# Video Moment Finder

## Project Summary

Video Moment Finder helps users turn a video into a small set of recommended photos or thumbnails. Instead of reviewing every frame manually, users can follow a target subject and review selected images with their timestamps and frame numbers.

Our initial audience is people choosing images from walking, sports, or performance videos. The current prototype focuses on human subjects and ranks frames by whole-image aesthetics.

## Current Implementation

**Video → YOLO person detection → ByteTrack tracking → Candidate sampling → Aesthetic scoring → Frame selection → Export**

1. **Detect people:** YOLO detects person bounding boxes in each frame.
2. **Follow a target:** ByteTrack associates detections across frames using motion prediction and bounding-box overlap. The default target is the largest person at the first tracked detection. Users can also specify a known tracking ID with `--target-id`.
3. **Collect candidates:** Save an original, full-resolution frame approximately every 0.5 seconds while the target ID is detected. Tracking still runs on every frame.
4. **Score images:** Aesthetic Predictor V2.5 scores the entire original frame. Detection boxes are excluded from scoring. The current ranking does not include separate sharpness, visibility, or composition scores.
5. **Select moments:** Rank candidates by aesthetic score and greedily select up to five, with at least one second between selected frames.
6. **Export results:** Save original images, target-box copies, CSV records, and an HTML gallery. The gallery includes scores, timestamps, frame numbers, and target IDs, with a “Show target box” option.

| Output | Purpose |
| --- | --- |
| `candidates/` | Original sampled frames |
| `candidates_target/` | Candidate copies with the target box and ID |
| `best/` | Selected original frames |
| `best_target/` | Selected copies with the target box and ID |
| `scores.csv` / `best_frames.csv` | Aesthetic scores, frame indices, timestamps, and image paths |
| `tracks.csv` | Per-frame tracking IDs, bounding boxes, confidence, and target flag |
| `summary.json` | Settings, counts, and completion status |
| `index.html` | Gallery for reviewing selected moments |

Frame numbers start at zero. Timestamps are calculated as frame index divided by FPS, assuming constant frame rate.

## Baseline and Development Iteration

**Baseline:** The original `human_detector.py` detects people and exports images or videos with bounding boxes. It does not maintain person identities, score frames, or select photos. A test image returned six person detections; an anime image returned zero.

**Iteration:** We connected person tracking, candidate sampling, aesthetic scoring, and photo export into an end-to-end pipeline. We then added target-box copies to make it easier to inspect which person the system followed.

## Results

These are observed prototype runs, not a quantitative benchmark of tracking accuracy or photo-selection quality.

| Test | Target ID | Candidate Frames | Selected Frames | Observation |
| --- | --- | --- | --- | --- |
| Single-main-person video, about 15 seconds | 1 | 30 | 5 | Complete pipeline ran successfully |
| Crowded video, about 6.34 seconds, automatic target | 3 | 1 | 1 | Target ID last appeared at about 0.40 seconds, before a second sample could be collected |
| Same crowded video, manually specified target | 6 | 9 | 5 | More candidates were collected from a longer-lasting track; target-box images allow visual inspection |

In the single-main-person run, the highest-ranked candidate was at **5.50 seconds, frame 132**, with an aesthetic score of approximately **5.887**.

The crowded-video experiment shows that initial target selection affects the results. A target can leave the frame or lose its ID, resulting in too few candidates. Producing five images does not prove that the tracker maintained the correct identity throughout the clip.

## Limitations and Next Steps

- **More scoring metrics:** Whole-frame aesthetics may favor an attractive background even when the target is blurry, small, or occluded. Add subject sharpness, visibility, and composition metrics.
- **User-controlled priorities:** Normalize the metrics to comparable scales, then combine them with a weighted average. Let users adjust the weights.
- **Target selection:** Replace manual ID entry with a preview where users can click the person they want to follow.
- **Tracking robustness:** ByteTrack does not use clothing color or appearance embeddings. Crowds, crossings, and occlusions can cause ID switches or lost targets. Evaluate appearance-assisted tracking and add lost-target feedback.
- **Frame diversity:** One-second spacing reduces adjacent picks but does not guarantee different-looking images. Add image-feature similarity filtering or cluster candidates by appearance/action before selecting high-scoring representatives.
- **Other subjects:** Consider pets, vehicles, and other supported detection classes after validating the human-subject workflow.
- **Evaluation:** Compare selections with human preferences and measure target continuity, selection quality, and redundancy across multiple videos.

## Demo Flow

1. Show the original single-person video and explain the goal.
2. Show the baseline detection image or video.
3. Explain the pipeline, emphasizing the difference between detection and tracking.
4. Open the single-person gallery and point out scores, timestamps, and frame numbers.
5. Show the crowded-video results: one candidate for the automatic target versus nine candidates for ID 6.
6. Expand “Show target box” to identify the tracked person.
7. Explain the planned metrics, user controls, tracking improvements, and diversity filtering.

## Run the Demo

Run these commands from the repository folder with the virtual environment activated:

```bash
# Single-main-person example
python moment_finder.py test.mp4

# Crowded example with the known target ID
python moment_finder.py crowd.mp4 --target-id 6 --max-seconds 10
```

Each run creates a new results folder under `output/`. Open its `index.html` in a browser or use prepared results during the presentation.
