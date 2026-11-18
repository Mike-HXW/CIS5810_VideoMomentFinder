# Video Moment Finder

Find still frames in a video. YOLO26 detects people, while BoT-SORT with ReID
tracks them for subject selection. Aesthetic Predictor V2.5 scores each
candidate's overall look; subject rankings also consider sharpness,
visibility, and composition.

## Setup (Windows)

```powershell
git clone https://github.com/Mike-HXW/CIS5810_VideoMomentFinder.git
cd CIS5810_VideoMomentFinder
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

To use an editable local copy of Aesthetic Predictor V2.5:

```powershell
git clone https://github.com/discus0434/aesthetic-predictor-v2-5.git external/aesthetic-predictor-v2-5
python -m pip install -e .\external\aesthetic-predictor-v2-5
```

The first run may download model weights. `imageio-ffmpeg` provides FFmpeg for
the thumbnail baseline on Windows.

## Command line

Put `test.mp4` in `input/` or pass a path to your video.

```powershell
python moment_finder.py test.mp4 --top-k 5
python human_detector.py test.mp4
```

`moment_finder.py` ranks frames. Use `--ranking-preset clear-subject` to give
sharpness more weight, or `--help` for the other presets and settings.
`human_detector.py` draws person boxes on an image or video.
The command-line ranking defaults to tracking the largest person at the first
detection. Tracking settings are in `botsort_reid.yaml`; ReID uses YOLO features
when compatible (`with_reid: True`, `model: auto`).

## Outputs and baselines

Each ranking run writes to `output/<video>_moments_<timestamp>/` by default.
Open `index.html` for the ranked gallery, or find the selected images in
`best/` and their scores in `best_frames.csv`.

- `baselines/aesthetic_only/top_1.png` is the tracked candidate with the highest
  raw aesthetic score. It reuses the scores computed for the main ranking.
- `baselines/ffmpeg_thumbnail/top_1.png` is a representative frame selected
  from samples across the full video, even when the main run has a time limit.

`summary.json` records the baseline selections.

## Gradio app

From the project folder with `.venv` active, run `python app.py` and open the
URL printed in the terminal (normally `http://127.0.0.1:7860`).

Upload a video or select one from `input/`. After the subject scan, choose a
person or keep **Entire scene**, then click **Analyze video**. For a selected
person, choose **Balanced**, **Clear subject**, **Subject fully shown**, or
**Best-looking scene**. The gallery shows frames, timestamps, scores, and the
output folder.
