"""Target-aware, model-free metrics for ranking video-frame candidates."""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from pathlib import Path

import cv2
import numpy as np


PRESET_WEIGHTS: dict[str, dict[str, float]] = {
    "balanced": {
        "aesthetic": 0.50,
        "sharpness": 0.30,
        "visibility": 0.15,
        "composition": 0.05,
    },
    "clear-subject": {
        "aesthetic": 0.35,
        "sharpness": 0.40,
        "visibility": 0.20,
        "composition": 0.05,
    },
    "subject-fully-shown": {
        "aesthetic": 0.25,
        "sharpness": 0.25,
        "visibility": 0.45,
        "composition": 0.05,
    },
    "best-looking-scene": {
        "aesthetic": 0.70,
        "sharpness": 0.15,
        "visibility": 0.10,
        "composition": 0.05,
    },
    # Diagnostic baseline used for comparisons; not intended as a UI preset.
    "aesthetic-only": {
        "aesthetic": 1.00,
        "sharpness": 0.00,
        "visibility": 0.00,
        "composition": 0.00,
    },
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _validated_box(
    bbox: Sequence[float], frame_width: int, frame_height: int,
) -> tuple[int, int, int, int] | None:
    if frame_width <= 0 or frame_height <= 0 or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
    x1 = max(0, min(frame_width, x1))
    x2 = max(0, min(frame_width, x2))
    y1 = max(0, min(frame_height, y1))
    y2 = max(0, min(frame_height, y2))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return x1, y1, x2, y2


def tenengrad_sharpness(frame: np.ndarray, bbox: Sequence[float]) -> float:
    """Measure subject-crop sharpness using Sobel gradient energy.

    The crop is resized to a common maximum dimension before convolution so
    scores are less sensitive to changing target-box size. A small Gaussian
    blur suppresses sensor/compression noise that would otherwise look sharp.
    """
    if frame is None or frame.ndim not in (2, 3):
        return 0.0
    height, width = frame.shape[:2]
    box = _validated_box(bbox, width, height)
    if box is None:
        return 0.0
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if min(crop.shape[:2]) < 8:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    target_long_side = 320
    scale = target_long_side / max(gray.shape[:2])
    resized_width = max(8, round(gray.shape[1] * scale))
    resized_height = max(8, round(gray.shape[0] * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    gray = cv2.resize(gray, (resized_width, resized_height), interpolation=interpolation)
    gray = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)

    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    score = float(np.mean(gradient_x * gradient_x + gradient_y * gradient_y))
    return score if math.isfinite(score) else 0.0


def visibility_components(
    bbox: Sequence[float],
    frame_width: int,
    frame_height: int,
    track_stability: float,
) -> dict[str, float]:
    """Return interpretable framing proxies for target visibility.

    This intentionally does not claim to detect semantic occlusion. It scores
    whether the tracked target is large enough, clear of frame boundaries, and
    present consistently in nearby frames.
    """
    box = _validated_box(bbox, frame_width, frame_height)
    if box is None:
        return {
            "area_ratio": 0.0,
            "size_score": 0.0,
            "edge_clearance_score": 0.0,
            "track_stability_score": _clamp01(track_stability),
            "visibility_score": 0.0,
        }

    x1, y1, x2, y2 = box
    area_ratio = ((x2 - x1) * (y2 - y1)) / float(frame_width * frame_height)
    # Targets below 2% of the frame are difficult to judge; 15% reaches full credit.
    size_score = _clamp01((area_ratio - 0.02) / (0.15 - 0.02))

    margins = (x1, y1, frame_width - x2, frame_height - y2)
    relative_clearance = min(margins) / float(min(frame_width, frame_height))
    # Three percent of the shorter frame side is enough to count as clear of an edge.
    edge_clearance_score = _clamp01(relative_clearance / 0.03)
    stability_score = _clamp01(track_stability)
    visibility_score = (
        0.50 * size_score
        + 0.35 * edge_clearance_score
        + 0.15 * stability_score
    )
    return {
        "area_ratio": area_ratio,
        "size_score": size_score,
        "edge_clearance_score": edge_clearance_score,
        "track_stability_score": stability_score,
        "visibility_score": _clamp01(visibility_score),
    }


def composition_score(
    bbox: Sequence[float], frame_width: int, frame_height: int,
) -> float:
    """Return a low-stakes rule-of-thirds proximity score in [0, 1]."""
    box = _validated_box(bbox, frame_width, frame_height)
    if box is None:
        return 0.0
    x1, y1, x2, y2 = box
    center_x = ((x1 + x2) / 2.0) / frame_width
    center_y = ((y1 + y2) / 2.0) / frame_height
    thirds = ((1 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1 / 3), (2 / 3, 2 / 3))
    distance = min(math.hypot(center_x - tx, center_y - ty) for tx, ty in thirds)
    sigma = 0.28
    return _clamp01(math.exp(-0.5 * (distance / sigma) ** 2))


def robust_normalize(values: Iterable[float]) -> list[float]:
    """Normalize values to [0, 1] using the 5th and 95th percentiles."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return []
    finite = np.isfinite(array)
    if not finite.any():
        return [0.0] * len(array)
    replacement = float(np.median(array[finite]))
    array = np.where(finite, array, replacement)
    lower, upper = np.percentile(array, [5.0, 95.0])
    if not math.isfinite(lower) or not math.isfinite(upper) or upper - lower < 1e-12:
        return [0.5] * len(array)
    normalized = np.clip((array - lower) / (upper - lower), 0.0, 1.0)
    return [float(value) for value in normalized]


def track_stability_for_frame(
    target_presence: Sequence[bool], frame_index: int, window_frames: int,
) -> float:
    """Fraction of frames containing the target around a candidate frame."""
    if not target_presence:
        return 0.0
    radius = max(1, int(window_frames))
    start = max(0, int(frame_index) - radius)
    end = min(len(target_presence), int(frame_index) + radius + 1)
    window = target_presence[start:end]
    return sum(bool(value) for value in window) / len(window) if window else 0.0


def score_candidates(
    candidates: list[dict],
    output_dir: str | Path,
    frame_width: int,
    frame_height: int,
    target_presence: Sequence[bool],
    fps: float,
    preset: str = "balanced",
) -> dict[str, float]:
    """Populate target-aware metric fields and final ``score`` in place."""
    if preset not in PRESET_WEIGHTS:
        raise ValueError(f"Unknown ranking preset: {preset}")
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive.")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be positive and finite.")

    output_dir = Path(output_dir)
    stability_window_frames = max(1, round(fps * 0.5))
    for candidate in candidates:
        image_path = output_dir / candidate["image"]
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise OSError(f"Cannot read candidate image for metrics: {image_path}")
        bbox = [candidate[key] for key in ("x1", "y1", "x2", "y2")]
        candidate["sharpness_raw"] = tenengrad_sharpness(frame, bbox)
        stability = track_stability_for_frame(
            target_presence, candidate["frame_index"], stability_window_frames,
        )
        candidate.update(visibility_components(bbox, frame_width, frame_height, stability))
        candidate["composition_score"] = composition_score(bbox, frame_width, frame_height)

    aesthetic_values = [candidate["aesthetic_score"] for candidate in candidates]
    sharpness_values = [candidate["sharpness_raw"] for candidate in candidates]
    aesthetic_normalized = robust_normalize(aesthetic_values)
    sharpness_normalized = robust_normalize(sharpness_values)
    weights = PRESET_WEIGHTS[preset]
    for candidate, aesthetic_score, sharpness_score in zip(
        candidates, aesthetic_normalized, sharpness_normalized,
    ):
        candidate["aesthetic_normalized"] = aesthetic_score
        candidate["sharpness_score"] = sharpness_score
        candidate["score"] = sum((
            weights["aesthetic"] * aesthetic_score,
            weights["sharpness"] * sharpness_score,
            weights["visibility"] * candidate["visibility_score"],
            weights["composition"] * candidate["composition_score"],
        ))
    return dict(weights)
