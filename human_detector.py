"""Human detection for images and videos using Ultralytics YOLO."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_DIR / "input"
OUTPUT_DIR = PROJECT_DIR / "output"

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}


@dataclass(frozen=True)
class Detection:
    """A detected person in pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class HumanDetector:
    """Detect people and draw bounding boxes around them."""

    def __init__(
        self,
        model: str = "yolo26n.pt",
        confidence: float = 0.25,
        device: str | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        self.model = YOLO(model)
        self.confidence = confidence
        self.device = device
        self.person_class_id = self._find_person_class_id(self.model.names)

    @staticmethod
    def _find_person_class_id(names: dict[int, str] | list[str]) -> int:
        items = names.items() if isinstance(names, dict) else enumerate(names)
        for class_id, name in items:
            if name.lower() == "person":
                return int(class_id)
        raise ValueError("The selected model does not contain a 'person' class")

    def detect_frame(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, list[Detection]]:
        """Return a copy of a frame with person boxes and its detections."""
        result = self.model.predict(
            source=frame,
            classes=[self.person_class_id],
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        if result.boxes is not None:
            coordinates = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            for coordinates_for_box, score in zip(coordinates, confidences):
                x1, y1, x2, y2 = (int(value) for value in coordinates_for_box)
                detections.append(Detection(x1, y1, x2, y2, float(score)))

        return self.draw_detections(frame, detections), detections

    @staticmethod
    def draw_detections(
        frame: np.ndarray, detections: Sequence[Detection]
    ) -> np.ndarray:
        """Draw the current box representation on a copy of a frame."""
        annotated = frame.copy()
        for detection in detections:
            top_left = (detection.x1, detection.y1)
            bottom_right = (detection.x2, detection.y2)
            cv2.rectangle(annotated, top_left, bottom_right, (0, 255, 0), 2)
            label = f"person {detection.confidence:.2f}"
            label_y = max(detection.y1 - 8, 20)
            cv2.putText(
                annotated,
                label,
                (detection.x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def process_image(self, input_path: Path, output_path: Path) -> int:
        """Detect people in an image, save it, and return the detection count."""
        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"Could not read image: {input_path}")

        annotated, detections = self.detect_frame(image)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), annotated):
            raise OSError(f"Could not write image: {output_path}")
        return len(detections)

    def process_video(self, input_path: Path, output_path: Path) -> int:
        """Detect people in every video frame and return the total box count."""
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if width <= 0 or height <= 0:
            capture.release()
            raise ValueError(f"Video has an invalid frame size: {input_path}")
        if fps <= 0:
            fps = 30.0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        codec = "XVID" if output_path.suffix.lower() == ".avi" else "mp4v"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise OSError(f"Could not create video: {output_path}")

        detection_count = 0
        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                annotated, detections = self.detect_frame(frame)
                writer.write(annotated)
                detection_count += len(detections)
        finally:
            capture.release()
            writer.release()

        return detection_count

    def process(self, input_path: str | Path, output_path: str | Path | None = None) -> Path:
        """Read bare filenames from input/ and save to output/ by default."""
        source = Path(input_path)
        if not source.is_absolute() and source.parent == Path("."):
            source = INPUT_DIR / source
        if not source.is_file():
            raise FileNotFoundError(f"Input file does not exist: {source}")

        suffix = source.suffix.lower()
        if suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported input file type: {suffix or '(none)'}")

        if output_path is None:
            output_suffix = source.suffix if suffix in IMAGE_EXTENSIONS else ".mp4"
            destination = OUTPUT_DIR / f"{source.stem}_detected{output_suffix}"
        else:
            destination = Path(output_path)

        if suffix in IMAGE_EXTENSIONS:
            if destination.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError("Image output must use a supported image extension")
        elif destination.suffix.lower() not in {".avi", ".mp4"}:
            raise ValueError("Video output must use an .avi or .mp4 extension")

        if source.resolve() == destination.resolve():
            raise ValueError("Output path must be different from the input path")

        if suffix in IMAGE_EXTENSIONS:
            count = self.process_image(source, destination)
            print(f"Detected {count} person(s).")
        else:
            count = self.process_video(source, destination)
            print(f"Drew {count} person box(es) across all video frames.")

        return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect people in an image or video with YOLO."
    )
    parser.add_argument(
        "input", help="Filename inside input/, or an explicit image/video path"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path (default: output/<input name>_detected.<extension>)",
    )
    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="YOLO model name or local weights path (default: yolo26n.pt)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum detection confidence from 0 to 1 (default: 0.25)",
    )
    parser.add_argument(
        "--device",
        help="Inference device, such as cpu, 0, or cuda:0 (default: automatic)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    detector = HumanDetector(
        model=args.model,
        confidence=args.confidence,
        device=args.device,
    )
    output = detector.process(args.input, args.output)
    print(f"Saved result to {output}")


if __name__ == "__main__":
    main()
