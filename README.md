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

### GPU (optional)

On Windows, pip installs the CPU-only PyTorch build, so the GPU options in the
app cannot be used. With an NVIDIA GPU, install the CUDA build after the
requirements. Pick the index URL that matches your driver at
<https://pytorch.org/get-started/locally/>; this project was tested with CUDA 13.0:

```powershell
python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.cuda.is_available())"
```

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

- `best/` and `best_target/` are full-quality PNGs of the selected frames; the
  target box is drawn only on these final picks. The sampled frames in
  `candidates/` are JPEG previews to save disk space; all scores are computed
  from the decoded frames, not from these previews.

- `baselines/aesthetic_only/top_1.png` is the tracked candidate with the highest
  raw aesthetic score. It reuses the scores computed for the main ranking.
- `baselines/ffmpeg_thumbnail/top_1.png` is a representative frame selected
  from samples across the full video, even when the main run has a time limit.

`summary.json` records the baseline selections.

## Gradio app

From the project folder with `.venv` active, run `python app.py` and open the
URL printed in the terminal (normally `http://127.0.0.1:7860`).

Upload a video or select one from `input/`. To rank the whole frame, keep
**Entire scene** and click **Analyze video**; no person scan is needed. To follow
a person, click **Find people**, pick a person, then click **Analyze video**.

For a selected person, choose **Balanced**, **Clear subject**, **Subject fully
shown**, or **Best-looking scene**. The gallery shows frames, timestamps, scores,
and the output folder.

The tracker keeps a lost person's ID for `track_buffer` frames (`botsort_reid.yaml`)
and compensates for camera motion, which helps with short detection gaps and small
pans. A person who walks out of the frame and comes back usually gets a new ID (see
[Future work](#future-work-identity-across-hard-cuts-and-re-entries)). If one person
shows up as several entries, click all of them to treat them as one person (click
again to remove).

## Tests and checks

```powershell
python -m unittest discover -s tests
python scripts/check_aesthetic.py path\to\image.jpg
```

`scripts/check_aesthetic.py` is a manual check that prints the aesthetic score
of one image (default `input/person.jpg`); it loads the full model.

## Future work: identity across hard cuts and re-entries

The app targets unedited clips that users record themselves, so this is not a current
goal. It matters for **edited videos** (hard cuts between shots) and, partly, for
**people who walk out of the frame and come back**. Everything below was prototyped and
measured in November 2026, then left out because it slowed the person scan by 4-5x
for little gain on the target videos. Plan B (select several entries as one person)
is the only part that shipped.

### The problem

- **Hard cut, same spot, different person.** In `top_gear.mp4` (4 hard cuts) the
  tracker kept ID 1 across every cut, so "Person 1" mixed Clarkson and Hammond.
  Close-up shots fill the frame, so the boxes before and after a cut overlap heavily
  (IoU 0.32-0.87) and BoT-SORT links them by position.
- **Re-entry.** A person who walks out and back in gets a new ID even with
  `track_buffer: 120`: the lost track's Kalman prediction keeps moving in the exit
  direction, so the new box no longer overlaps it, and ReID is only consulted when
  boxes overlap.

### Why tracker settings alone cannot fix it

From `ultralytics/trackers/bot_sort.py` (`get_dists`), version 8.4.158:

- The match cost is `min(position cost, appearance cost)`. Appearance can only make a
  match easier; it can never veto a match that position already allows.
- The appearance cost is ignored when IoU < `proximity_thresh` (0.5), so ReID cannot
  re-find someone in a different place.
- `match_thresh` can only reject by position. On `top_gear` the two wrong links
  (Clarkson <-> Hammond) cost 0.17-0.19, the same as normal same-person links (median
  0.11, 95% under 0.23). Rejecting them needs 0.15, which breaks 27% of correct links.
- `track_buffer` and GMC do not help at a hard cut: there is no gap frame, and the
  motion estimate across a cut is meaningless.

### Appearance features measured at the 4 cuts of `top_gear.mp4`

Distance between the boxes on both sides of each cut (lower = more alike):

| Cut | Truth | Colour histogram | BoT-SORT ReID (`model: auto`) | SigLIP |
|---|---|---|---|---|
| 62 | Clarkson close-up -> wide (same) | 0.48 | 0.59 | 0.145 |
| 289 | Clarkson wide -> close-up (same) | 0.43 | 0.71 | 0.162 |
| 345 | Clarkson -> Hammond (different) | 0.50 | 0.04 | 0.184 |
| 420 | Hammond -> Clarkson (different) | 0.43 | 0.03 | 0.227 |

- **Colour histogram**: a change of framing looks as different as a change of person.
- **ReID with `model: auto`** (YOLO's own features) is inverted: it compares the
  framing, not the person. Turning ReID off changed tracking results very little.
- **SigLIP** (the vision tower inside the aesthetic model, so no new dependency) is
  the only one in the right order, with a small margin.

Averaging SigLIP over 5 crops per track piece separated people better: Clarkson's 4
pieces were 0.04-0.14 apart. But same-person and different-person distances overlap
from about 0.09 (`top_gear`: Clarkson close-up vs wide 0.117; `mercedes.mp4`: two
different women 0.093), and small, dark, blurry crops all look alike to SigLIP
(distances 0.02-0.06 between unrelated people).

### The design that was prototyped

1. **Hard-cut detection.** Store a 96x96 grayscale thumbnail of every frame while
   tracking. Crop away letterbox bars (rows/columns with mean brightness <= 8; fall
   back to the full frame for very dark videos). A cut is a mean absolute frame
   difference above `max(6 x median difference, 20)`. Found all 4 cuts in `top_gear`
   (normal frames max 15.9, cuts 38+) and 0 false cuts in the one-shot `kongfu.mp4`;
   `mercedes.mp4` was borderline (normal frames up to 28.0, cuts from 29.9).
2. **Split tracks at cuts.** Each (tracker ID, shot) becomes its own entry with a new
   ID, so a cut can never join two people. This alone fixes the wrong-person problem
   and costs about 2-3 s (a second decode-only pass for preview crops).
3. **Group pieces into people with SigLIP.** Embed 5 crops per piece, then
   average-linkage merge pieces that are never on screen at the same time. Only pieces
   with >= 10 frames and a box >= 2% of the frame take part. A threshold of 0.07 gave
   no wrong merges on `top_gear`, `mercedes` and `kongfu`, but merged little
   (Clarkson's 3 close-ups; one person in `mercedes`). A wrong merge is worse than a
   missed one, because the UI can combine entries but cannot split them.
4. **Plan B (shipped).** Selecting several entries treats them as one person; `run()`
   accepts a list of IDs.

An alternative that was designed but not built: subclass the BoT-SORT tracker and veto
a match at a cut when appearance disagrees, with a larger `track_buffer` so the old
track can be re-found later (e.g. close-up -> other person -> same close-up). It only
works where the person returns to the same place, and the histogram threshold on
`top_gear` had a margin of 0.11.

### Cost that blocked it

Measured on GPU:

| | `top_gear` | `mercedes` |
|---|---|---|
| Scan before | 18.7 s | 44.4 s |
| With cut splitting only | 21.3 s | 42.2 s |
| With splitting + SigLIP grouping | 31.4 s | 73.6 s |
| Extra on the first scan: loading SigLIP | +35 s | +35 s |

### If this is picked up again

- **Cut splitting** (step 1-2) is cheap and safe; add it first for edited videos.
- **Make SigLIP cheap before using it**: embed only pieces that can be grouped
  (most pieces in `mercedes` are tiny), embed all crops in a few batches instead of
  per piece, and load the model in the background when the app starts.
- **A real person re-identification model** (e.g. OSNet) or face embeddings should
  separate people much better than SigLIP, at the cost of a new dependency.
- **Re-entry in unedited clips** is the same grouping problem without cuts: group
  tracks that are never on screen together by appearance.
- **Test clips to collect**: edited videos with cuts between people standing in the
  same spot, several people in similar clothes, and a person walking out and back in.
