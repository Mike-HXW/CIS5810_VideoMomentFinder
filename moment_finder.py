"""Track one person and rank video frames with target-aware quality metrics."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import torch
from PIL import Image

from human_detector import HumanDetector, INPUT_DIR, OUTPUT_DIR, PROJECT_DIR
from target_metrics import PRESET_WEIGHTS, score_candidates


def select_best(candidates, top_k, min_gap):
    """Greedy final-score ranking with minimum temporal separation."""
    selected = []
    for candidate in sorted(candidates, key=lambda c: (-c['score'], c['frame_index'])):
        if all(abs(candidate['timestamp_seconds'] - c['timestamp_seconds']) >= min_gap
               for c in selected):
            selected.append(candidate)
            if len(selected) == top_k:
                break
    return selected


def save_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def draw_target(frame, target):
    """Annotate a copy; never modify the original frame used for scoring."""
    annotated = frame.copy()
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(target[k]) for k in ('x1', 'y1', 'x2', 'y2'))
    x1, x2 = max(0, min(width - 1, x1)), max(0, min(width - 1, x2))
    y1, y2 = max(0, min(height - 1, y1)), max(0, min(height - 1, y2))
    thickness = max(2, round(min(width, height) / 300))
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), thickness)
    label = f'Target ID {target["track_id"]}'
    scale = max(0.5, min(width, height) / 1100)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    tx = max(0, min(x1, width - tw - 8))
    ty = min(height - baseline - 4, max(th + 4, y1 - 8))
    cv2.rectangle(annotated, (tx, ty - th - 4),
                  (min(width - 1, tx + tw + 8), ty + baseline + 4), (0, 0, 0), -1)
    cv2.putText(annotated, label, (tx + 4, ty), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 255, 0), thickness, cv2.LINE_AA)
    return annotated


class AestheticScorer:
    def __init__(self, device):
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
        print('Loading aesthetic model (first run may download weights)...', flush=True)
        self.device = device
        self.model, self.processor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True, trust_remote_code=True,
        )
        self.model = self.model.to(device=device, dtype=torch.float32).eval()

    def score(self, path):
        with Image.open(path) as image:
            pixels = self.processor(images=image.convert('RGB'), return_tensors='pt').pixel_values
        with torch.inference_mode():
            score = self.model(pixels.to(self.device, dtype=torch.float32)).logits.squeeze().item()
        if not math.isfinite(score):
            raise ValueError(f'Non-finite aesthetic score for {path}')
        return score


def run(args):
    source = Path(args.input)
    if not source.is_absolute() and source.parent == Path('.'):
        source = INPUT_DIR / source
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f'Video not found: {source}')
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f'Cannot open video: {source}')
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError('Video has no valid FPS; timestamps cannot be calculated.')
    frame_width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_width <= 0 or frame_height <= 0:
        capture.release()
        raise ValueError('Video has no valid frame dimensions.')
    output = Path(args.output) if args.output else OUTPUT_DIR / (
        f'{source.stem}_moments_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    try:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'candidates').mkdir()
        (output / 'best').mkdir()
        (output / 'candidates_target').mkdir()
        (output / 'best_target').mkdir()
        print(f'Results folder: {output.resolve()}', flush=True)
        model_path = Path(args.model)
        if not model_path.is_absolute() and (PROJECT_DIR / model_path).is_file():
            model_path = PROJECT_DIR / model_path
        detector = HumanDetector(model=str(model_path), confidence=args.confidence, device=args.device)
        target_id = args.target_id
        candidates = []
        target_presence = []
        frame_index = 0
        target_frames = 0
        next_sample_time = 0.0
        with (output / 'tracks.csv').open('w', newline='', encoding='utf-8') as handle:
            fields = ['frame_index', 'timestamp_seconds', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence', 'is_target']
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            print('Tracking every frame with ByteTrack...', flush=True)
            while True:
                timestamp = frame_index / fps
                if args.max_seconds is not None and timestamp >= args.max_seconds:
                    break
                success, frame = capture.read()
                if not success:
                    break
                result = detector.model.track(
                    source=frame, persist=True, tracker='bytetrack.yaml',
                    classes=[detector.person_class_id], conf=args.confidence,
                    device=args.device, verbose=False,
                )[0]
                people = []
                if result.boxes is not None and result.boxes.id is not None:
                    for box, track_id, confidence in zip(
                        result.boxes.xyxy.cpu().tolist(), result.boxes.id.int().cpu().tolist(),
                        result.boxes.conf.cpu().tolist(),
                    ):
                        x1, y1, x2, y2 = map(int, box)
                        people.append(dict(frame_index=frame_index, timestamp_seconds=timestamp,
                                           track_id=track_id, x1=x1, y1=y1, x2=x2, y2=y2,
                                           confidence=confidence))
                if target_id is None and people:
                    target_id = max(people, key=lambda p: (p['x2']-p['x1'])*(p['y2']-p['y1']))['track_id']
                    print(f'Locked target ID {target_id} (largest person at first detection).', flush=True)
                target = None
                for person in people:
                    writer.writerow({**person, 'is_target': person['track_id'] == target_id})
                    if person['track_id'] == target_id:
                        target = person
                target_presence.append(target is not None)
                if target is not None:
                    target_frames += 1
                    if timestamp + 1e-9 >= next_sample_time:
                        filename = f'candidates/frame_{frame_index:07d}.png'
                        if not cv2.imwrite(str(output / filename), frame):
                            raise OSError(f'Cannot save {filename}')
                        target_filename = f'candidates_target/frame_{frame_index:07d}.png'
                        if not cv2.imwrite(str(output / target_filename), draw_target(frame, target)):
                            raise OSError(f'Cannot save {target_filename}')
                        candidates.append({**target, 'image': filename, 'target_image': target_filename})
                        next_sample_time = timestamp + args.sample_seconds
                frame_index += 1
                if frame_index % max(1, round(fps * 2)) == 0:
                    print(f'Tracked {timestamp:.1f}s; saved {len(candidates)} candidates.', flush=True)
    finally:
        capture.release()

    ranking_preset = getattr(args, 'ranking_preset', 'balanced')
    summary = dict(source=str(source), fps=fps, frame_width=frame_width,
                   frame_height=frame_height, frame_index_base=0,
                   timestamp_method='frame_index / fps (constant-frame-rate assumption)',
                   processed_frames=frame_index, target_id=target_id, target_visible_frames=target_frames,
                   sample_seconds=args.sample_seconds, min_gap_seconds=args.min_gap,
                   requested_top_k=args.top_k, candidate_count=len(candidates),
                   ranking_preset=ranking_preset,
                   selection='Target-aware weighted score with temporal separation',
                   status='tracking_complete')
    summary_path = output / 'summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if not candidates:
        raise ValueError(f'No candidate frames for target {target_id}. See {output / "tracks.csv"}.')

    scorer = AestheticScorer(args.aesthetic_device)
    for index, candidate in enumerate(candidates, 1):
        candidate['aesthetic_score'] = scorer.score(output / candidate['image'])
        print(f'Aesthetic {index}/{len(candidates)}: frame {candidate["frame_index"]}, '
              f'{candidate["timestamp_seconds"]:.2f}s, '
              f'raw {candidate["aesthetic_score"]:.3f}', flush=True)
    weights = score_candidates(
        candidates=candidates,
        output_dir=output,
        frame_width=frame_width,
        frame_height=frame_height,
        target_presence=target_presence,
        fps=fps,
        preset=ranking_preset,
    )
    for index, candidate in enumerate(candidates, 1):
        print(
            f'Target metrics {index}/{len(candidates)}: frame {candidate["frame_index"]}, '
            f'final {candidate["score"]:.3f}, sharp {candidate["sharpness_score"]:.3f}, '
            f'visible {candidate["visibility_score"]:.3f}, '
            f'composition {candidate["composition_score"]:.3f}',
            flush=True,
        )
    fields = [
        'frame_index', 'timestamp_seconds', 'track_id',
        'x1', 'y1', 'x2', 'y2', 'confidence', 'image', 'target_image',
        'aesthetic_score', 'aesthetic_normalized',
        'sharpness_raw', 'sharpness_score',
        'area_ratio', 'size_score', 'edge_clearance_score',
        'track_stability_score', 'visibility_score', 'composition_score', 'score',
    ]
    save_csv(output / 'scores.csv', sorted(candidates, key=lambda c: -c['score']), fields)
    best = select_best(candidates, args.top_k, args.min_gap)
    cards = []
    for rank, candidate in enumerate(best, 1):
        filename = f'best/{rank:02d}_frame_{candidate["frame_index"]:07d}.png'
        shutil.copy2(output / candidate['image'], output / filename)
        target_filename = f'best_target/{rank:02d}_frame_{candidate["frame_index"]:07d}.png'
        shutil.copy2(output / candidate['target_image'], output / target_filename)
        candidate = {**candidate, 'rank': rank, 'image': filename, 'target_image': target_filename}
        best[rank - 1] = candidate
        cards.append(f'<figure><a href="{filename}"><img src="{filename}" alt="Rank {rank}"></a>'
                     f'<figcaption>#{rank} · Final {candidate["score"]:.3f} · '
                     f'Aesthetic {candidate["aesthetic_normalized"]:.3f} · '
                     f'Sharpness {candidate["sharpness_score"]:.3f} · '
                     f'Visibility {candidate["visibility_score"]:.3f} · '
                     f'Composition {candidate["composition_score"]:.3f} · '
                     f'{candidate["timestamp_seconds"]:.2f}s · Frame {candidate["frame_index"]}'
                     f' · Person ID {target_id}</figcaption>'
                     f'<details><summary>Show target box</summary>'
                     f'<a href="{target_filename}"><img src="{target_filename}" alt="Target ID {target_id}"></a>'
                     f'</details></figure>')
    save_csv(output / 'best_frames.csv', best, ['rank'] + fields)
    (output / 'index.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Video Moment Finder</title>'
        '<style>body{font:16px system-ui;margin:32px;background:#f4f4f4} '
        'main{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}'
        'figure{margin:0;background:white;padding:12px}img{width:100%}figcaption{padding:12px 0}</style>'
        f'<h1>Best Video Moments</h1><p>{html.escape(source.name)} · Target ID {target_id}</p>'
        f'<p>Ranking preset: {html.escape(ranking_preset)}. '
        'Final score combines whole-frame aesthetics, subject sharpness, '
        'subject visibility, and low-weight composition. Frame numbers start at 0. '
        'Time separation reduces adjacent picks; it does not guarantee visual diversity.</p><main>'
        + ''.join(cards) + '</main></html>', encoding='utf-8',
    )
    summary.update(status='complete', selected_count=len(best), metric_weights=weights)
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f'Done. Selected {len(best)} of {len(candidates)} candidates.', flush=True)
    print(f'Open gallery: {output.resolve() / "index.html"}', flush=True)
    if len(best) < args.top_k:
        print('Fewer frames selected: limited target candidates and/or minimum time gap.', flush=True)
    return output


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Video filename in input/, or explicit path')
    parser.add_argument('--output', help='New output directory (must not already exist)')
    parser.add_argument('--model', default='yolo26n.pt')
    parser.add_argument('--device', default='cpu', help='YOLO device, e.g. cpu or mps')
    parser.add_argument('--aesthetic-device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--target-id', type=int, help='Specific track ID; otherwise lock first largest person')
    parser.add_argument('--sample-seconds', type=float, default=0.5)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--min-gap', type=float, default=1.0, help='Seconds between selected frames')
    parser.add_argument('--max-seconds', type=float, help='Process only the beginning of the video')
    parser.add_argument(
        '--ranking-preset', default='balanced', choices=sorted(PRESET_WEIGHTS),
        help='Fixed target-aware metric weights; aesthetic-only is a diagnostic baseline',
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    for name in ['sample_seconds', 'max_seconds']:
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f'--{name.replace("_", "-")} must be positive and finite')
    if not math.isfinite(args.min_gap) or args.min_gap < 0 or args.top_k < 1:
        parser.error('--min-gap must be nonnegative and finite; --top-k must be at least 1')
    if not 0 <= args.confidence <= 1:
        parser.error('--confidence must be between 0 and 1')
    try:
        run(args)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
