# Video Moment Finder web app

The local Gradio app supports two video sources:

- Upload a video. The app copies it into `input/` before analysis.
- Select an existing video from `input/` and refresh the list without restarting.

The app calls the same tracking and aesthetic-scoring pipeline as
`moment_finder.py`. Selected frames, timestamps, frame numbers, aesthetic
scores, and the output directory are shown after each run.

## Start

Activate the project virtual environment, then run:

```powershell
python app.py
```

Open the local URL printed by Gradio, normally `http://127.0.0.1:7860`.

CUDA-capable machines default to NVIDIA GPU device `0` for YOLO and `cuda`
for aesthetic scoring. CPU remains available from the device controls.
