# Video Moment Finder

Find the best still frames of a person in a video. YOLO and ByteTrack follow
the largest person in the first frame with a tracked detection. Aesthetic
Predictor V2.5 scores each candidate's overall look, and the ranking also
considers subject sharpness, visibility, and composition.

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

Upload a video or select one from `input/`. Choose **Balanced**, **Clear subject**,
**Subject fully shown**, or **Best-looking scene**, then click **Analyze video**.
The gallery shows the ranked frames, timestamps, scores, and output folder.
