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

A person who leaves the frame and comes back to about the same spot within
`track_buffer` frames (90, about 3 s at 30 fps) keeps the same ID; see
[Re-entry compensation](#re-entry-compensation-keeping-one-id-when-the-camera-turns-away).
If one person still shows up as several entries, click all of them to treat them as
one person (click again to remove).

## Re-entry compensation: keeping one ID when the camera turns away

A common way to film on a phone is to turn away from a person for a few seconds and
then turn back. Out of the box, the tracker gives that person a new ID every time,
so one person is split into several entries. `_freeze_lost_tracks()` in
`human_detector.py` fixes this. It patches the tracker when the module is imported,
so the person scan and `moment_finder.py` both use it. It adds no computation.

### How the tracker links people across frames

The tracker is BoT-SORT (`botsort_reid.yaml`), which is ByteTrack plus camera-motion
compensation (GMC) and appearance matching (ReID):

- **ByteTrack** predicts where each person will be in the next frame with a Kalman
  filter (box position, size and their velocities), then gives each new YOLO box the
  ID of the prediction it overlaps most. When nobody matches a track, the track goes
  on a "lost" list and can still be matched for `track_buffer` frames.
- **GMC** estimates how the camera moved between frames and moves every predicted
  box by the same amount.
- **ReID** compares appearance, but only when boxes already overlap (IoU >= 0.5), so
  it cannot find a person whose predicted box is in the wrong place.

So a returning person keeps the old ID only if the lost track's box still overlaps
where they reappear (IoU above about 0.2).

### What went wrong while the person was gone

Measured on `input/person_in_out.mp4` (412 frames, 30 fps, 720x1280): the person is
on screen three times and is gone for 71 and 81 frames in between. Ultralytics
8.4.158 had three problems:

| Problem | What happened in the clip |
|---|---|
| A lost track keeps moving at the speed it had when it disappeared. Leaving the frame is fast, so the box flies off. | After 71 lost frames the predicted box was at x ~ 2000 (the frame is 720 wide). |
| The first frame after the last detection is predicted one step further. A person leaving through the bottom edge looks like a box that is quickly getting shorter. | The box shrank from 121 px tall to 23 px, then stayed that way. |
| GMC keeps moving the lost box. The camera turns away and back, so the real net motion is about zero, but the per-frame errors add up over 80 frames. | The box drifted ~100 px and no longer overlapped the returning person (IoU 0). |

Each of these alone is enough to give the person a new ID.

### What the patch changes

Only lost tracks are affected. While a person is on screen, tracking works exactly
as before.

1. When a track is lost, its box goes back to the last box that matched a detection
   (this undoes the extra step).
2. While it is lost, its position and size do not change (all velocities are zero).
3. GMC is not applied to lost tracks.

In short: while a person is gone, the tracker assumes they are still where they
were last seen.

### Result on `person_in_out.mp4`

| Version | IDs for the person | Appearances joined |
|---|---|---|
| Ultralytics default | 3 | none |
| Patch steps 1-2 only, GMC on lost tracks | 2 | first return |
| Full patch, `track_buffer: 90` | 1 | both returns |

The clip also has two short extra IDs from duplicate YOLO boxes (one frame at frame
107, and a frame-wide box from frame 338). They are detection errors, not re-entries.

`track_buffer` must be longer than the time the person is gone: at 60 frames the
lost track is deleted before either return. Values much larger than needed make it
more likely that two people who cross get each other's IDs.

### Limits

- The person must come back to about the same place in the frame. If the camera does
  not return to where it was, or the person has moved, they get a new ID (as before
  the patch). Use multi-select to combine the entries.
- It does not help at hard cuts in edited videos; see
  [Future work](#future-work-identity-across-hard-cuts-and-re-entries).
- It relies on Ultralytics internals (`STrack.multi_predict`,
  `BOTrack.multi_predict`, `byte_tracker.multi_gmc`). Recheck it when
  `ultralytics==8.4.158` in `requirements.txt` is upgraded.

## Tests and checks

```powershell
python -m unittest discover -s tests
python scripts/check_aesthetic.py path\to\image.jpg
```

`scripts/check_aesthetic.py` is a manual check that prints the aesthetic score
of one image (default `input/person.jpg`); it loads the full model.

## Future work: identity across hard cuts and re-entries

The app targets unedited clips that users record themselves, so this is not a current
goal. It matters for **edited videos** (hard cuts between shots) and for **people who
come back somewhere else in the frame** (returning to the same spot is already handled
by [re-entry compensation](#re-entry-compensation-keeping-one-id-when-the-camera-turns-away)).
Everything below was prototyped and
measured in November 2026, then left out because it slowed the person scan by 4-5x
for little gain on the target videos. Plan B (select several entries as one person)
is the only part that shipped.

### The problem

- **Hard cut, same spot, different person.** In `top_gear.mp4` (4 hard cuts) the
  tracker kept ID 1 across every cut, so "Person 1" mixed Clarkson and Hammond.
  Close-up shots fill the frame, so the boxes before and after a cut overlap heavily
  (IoU 0.32-0.87) and BoT-SORT links them by position.
- **Re-entry somewhere else.** Re-entry compensation keeps a lost track where it was
  last seen. A person who comes back in a different place does not overlap it and
  gets a new ID, because ReID is only consulted when boxes overlap.

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
- **Re-entry in a different place** is the same grouping problem without cuts: group
  tracks that are never on screen together by appearance.
- **Test clips to collect**: edited videos with cuts between people standing in the
  same spot, several people in similar clothes, and a person who leaves and comes back
  somewhere else in the frame.
