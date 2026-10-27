# Video Moment Finder web app

The local Gradio app supports two video sources:

- Upload a video. The app copies it into `input/` before analysis.
- Select an existing video from `input/` and refresh the list without restarting.

The app calls the same target-aware ranking pipeline as `moment_finder.py`.
Choose one fixed ranking preference instead of tuning numeric weights:

- **Balanced**: a mix of whole-frame aesthetics, subject sharpness, and visibility.
- **Clear subject**: gives subject sharpness more influence.
- **Subject fully shown**: gives target size, edge clearance, and tracking stability more influence.
- **Best-looking scene**: gives whole-frame aesthetic quality more influence.

Selected frames, timestamps, component scores, and the output directory are
shown after each run.

## Start

Activate the project virtual environment, then run:

```powershell
python app.py
```

Open the local URL printed by Gradio, normally `http://127.0.0.1:7860`.

CUDA-capable machines default to NVIDIA GPU device `0` for YOLO and `cuda`
for aesthetic scoring. CPU remains available from the device controls.
