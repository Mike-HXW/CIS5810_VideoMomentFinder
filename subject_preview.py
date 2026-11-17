"""Scan a video once and prepare selectable person tracks."""

import cv2
import numpy as np

from human_detector import HumanDetector


def scan_subjects(source, model, device, confidence):
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    detector = HumanDetector(model=model, confidence=confidence, device=device)
    tracks = {}
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = detector.model.track(
            source=frame, persist=True, tracker='bytetrack.yaml',
            classes=[detector.person_class_id], conf=confidence, device=device, verbose=False,
        )[0]
        height, width = frame.shape[:2]
        if result.boxes is not None and result.boxes.id is not None:
            for box, track_id, score in zip(result.boxes.xyxy.cpu().tolist(),
                                             result.boxes.id.int().cpu().tolist(),
                                             result.boxes.conf.cpu().tolist()):
                x1, y1, x2, y2 = map(int, box)
                x1, x2 = max(0, x1), min(width, x2)
                y1, y2 = max(0, y1), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                area = (x2 - x1) * (y2 - y1) / (width * height)
                track = tracks.setdefault(track_id, {'frames': {}, 'areas': [], 'scores': [],
                                                      'preview_quality': -1})
                track['frames'][index] = {'frame_index': index, 'timestamp_seconds': index / fps,
                    'track_id': track_id, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'confidence': score}
                track['areas'].append(area)
                track['scores'].append(score)
                margin = min(x1, y1, width - x2, height - y2) / min(width, height)
                quality = area * score * (0.5 + min(0.5, margin / 0.06))
                if quality > track['preview_quality']:
                    crop = frame[y1:y2, x1:x2]
                    scale = min(1.0, 320 / max(crop.shape[:2]))
                    crop = cv2.resize(crop, None, fx=scale, fy=scale)
                    track['preview'] = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    track['preview_quality'] = quality
        index += 1
    capture.release()
    for track_id, track in tracks.items():
        track['id'] = track_id
        track['size'] = float(np.percentile(track.pop('areas'), 75))
        track['confidence'] = float(np.median(track.pop('scores')))
        track['seconds'] = len(track['frames']) / fps
        track['recommended'] = track['size'] >= 0.05 and track['seconds'] >= 1.0
        track.pop('preview_quality')
    ordered = sorted(tracks.values(), key=lambda t: (-t['recommended'], -t['size'], -t['seconds'], -t['confidence']))
    return {'source': str(source), 'tracks': ordered}
