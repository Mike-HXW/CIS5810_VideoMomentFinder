"""Local Gradio interface for Video Moment Finder."""
from __future__ import annotations

import argparse
import csv
import html
import math
import shutil
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch

from human_detector import INPUT_DIR, OUTPUT_DIR, PROJECT_DIR
from moment_finder import run


VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
PRESET_CHOICES = [
    ("Balanced", "balanced"),
    ("Clear subject", "clear-subject"),
    ("Subject fully shown", "subject-fully-shown"),
    ("Best-looking scene", "best-looking-scene"),
]
PRESET_LABELS = {value: label for label, value in PRESET_CHOICES}
SCORE_HEADERS = [
    "Rank", "Timestamp", "Frame", "Final", "Aesthetic",
    "Sharpness", "Visibility", "Composition",
]


APP_CSS = """
:root {
  --vmf-ink: #10213f;
  --vmf-muted: #64748b;
  --vmf-line: #dbe4f0;
  --vmf-panel: #f7faff;
  --vmf-accent: #0f67ff;
}
.gradio-container {
  max-width: 1500px !important;
  margin: 0 auto !important;
  padding: 28px 24px 36px !important;
  color: var(--vmf-ink) !important;
  --block-label-text-color: #000000 !important;
  --block-label-text-color-dark: #000000 !important;
}
#vmf-header { margin-bottom: 18px; }
#vmf-header h1 {
  color: var(--vmf-ink);
  font-size: clamp(2rem, 3vw, 3rem);
  letter-spacing: -0.045em;
  line-height: 1.05;
  margin: 0 0 8px;
}
#vmf-header p { color: var(--vmf-muted); font-size: 1.05rem; margin: 0; }
#control-panel, #results-panel {
  background: var(--vmf-panel);
  border: 1px solid var(--vmf-line);
  border-radius: 14px;
  padding: 16px;
}
#control-panel { align-self: flex-start; }
#control-panel .tabs { background: transparent; }
#ranking-preset { margin: 14px 0 4px; }
#ranking-preset label { color: #000000 !important; }
[role="listbox"] {
  background: #ffffff !important;
  border: 1px solid #cbd8e8 !important;
  color: var(--vmf-ink) !important;
}
[data-testid="dropdown-option"] {
  background: #ffffff !important;
  color: var(--vmf-ink) !important;
}
[data-testid="dropdown-option"].selected,
[data-testid="dropdown-option"]:hover {
  background: #eaf2ff !important;
  color: var(--vmf-ink) !important;
}
/* Gradio adds Tailwind's dark:bg-gray-600 to selected dropdown rows. */
html body .gradio-container ul.option-list li[data-testid="dropdown-option"].item.selected.active.bg-gray-100.dark\\:bg-gray-600,
html body .gradio-container ul.option-list li[data-testid="dropdown-option"].item {
  background: #eaf2ff !important;
  color: var(--vmf-ink) !important;
}
html body .gradio-container ul.option-list li[data-testid="dropdown-option"]:not(.selected) {
  background: #ffffff !important;
}
#control-panel label,
#control-panel label span,
#control-panel [data-testid="block-label"],
#control-panel [data-testid="block-label"] *,
#control-panel [data-testid="block-info"],
#control-panel .block-label,
#control-panel .label,
#results-panel label,
#results-panel label span,
#results-panel .block-label {
  color: #000000 !important;
  opacity: 1 !important;
}
#analyze-button {
  background: var(--vmf-accent) !important;
  border: 0 !important;
  color: #fff !important;
  min-height: 48px;
  font-size: 1rem;
  font-weight: 700;
}
#results-heading h2 { color: var(--vmf-ink); margin: 0 0 8px; }
.vmf-empty {
  border: 1px dashed #c8d5e6;
  border-radius: 12px;
  color: var(--vmf-muted);
  padding: 44px 24px;
  text-align: center;
}
.vmf-success {
  background: #eaf9f0;
  border: 1px solid #bce8cb;
  border-radius: 10px;
  color: #17653a;
  padding: 12px 14px;
}
.vmf-success strong { color: #0d512d; }
#moments-gallery { background: #fff; border-color: var(--vmf-line); }
#moments-gallery .caption-label {
  background: rgba(15, 23, 42, 0.92) !important;
  color: #ffffff !important;
}
#moments-table { background: #fff; }
.vmf-score-title { color: #000000; font-size: 0.9rem; margin: 0 0 8px; }
.vmf-score-table {
  border: 1px solid var(--vmf-line);
  border-collapse: separate;
  border-radius: 10px;
  border-spacing: 0;
  overflow: hidden;
  width: 100%;
  font-size: 0.88rem;
}
.vmf-score-table th, .vmf-score-table td {
  border-bottom: 1px solid var(--vmf-line);
  color: var(--vmf-ink);
  padding: 10px 12px;
  text-align: left;
}
.vmf-score-table th { background: #eef4fb; font-size: 0.82rem; }
.vmf-score-table td { background: #ffffff; }
.vmf-score-table tr:last-child td { border-bottom: 0; }
@media (max-width: 900px) {
  .vmf-score-table { display: block; overflow-x: auto; white-space: nowrap; }
}
@media (max-width: 900px) {
  .gradio-container { padding: 18px 12px 28px !important; }
  #control-panel, #results-panel { padding: 12px; }
}
"""


def list_input_videos() -> list[str]:
    """Return input-folder video names in a stable, user-friendly order."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        (path.name for path in INPUT_DIR.iterdir()
         if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
        key=str.casefold,
    )


def refresh_input_choices():
    choices = list_input_videos()
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def _safe_uploaded_name(upload_path: Path) -> str:
    stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in upload_path.stem)
    stem = stem.strip("_") or "uploaded_video"
    suffix = upload_path.suffix.lower() if upload_path.suffix.lower() in VIDEO_EXTENSIONS else ".mp4"
    return f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{suffix}"


def _resolve_source(source_mode: str, uploaded_video: str | None, input_video: str | None) -> Path:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    if source_mode == "upload":
        if not uploaded_video:
            raise ValueError("Upload a video before starting the analysis.")
        uploaded_path = Path(uploaded_video)
        if not uploaded_path.is_file():
            raise FileNotFoundError(f"Uploaded video is unavailable: {uploaded_path}")
        destination = INPUT_DIR / _safe_uploaded_name(uploaded_path)
        shutil.copy2(uploaded_path, destination)
        return destination.resolve()

    if not input_video:
        raise ValueError("Choose a video from the input folder before starting the analysis.")
    source = (INPUT_DIR / input_video).resolve()
    input_root = INPUT_DIR.resolve()
    if source.parent != input_root or not source.is_file():
        raise ValueError("The selected input-folder video is invalid.")
    return source


def _optional_positive(value, label: str) -> float | None:
    if value in (None, ""):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be positive.")
    return number


def _load_results(output: Path):
    csv_path = output / "best_frames.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Result table not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    gallery = []
    table = []
    for row in rows:
        rank = int(row["rank"])
        timestamp = float(row["timestamp_seconds"])
        frame_index = int(row["frame_index"])
        final_score = float(row["score"])
        image_path = output / row["image"]
        caption = (
            f"#{rank}  ·  {timestamp:.2f}s  ·  Frame {frame_index}  ·  "
            f"Final {final_score:.3f}"
        )
        gallery.append((str(image_path), caption))
        def metric(name: str) -> str:
            value = row.get(name)
            return f"{float(value):.3f}" if value not in (None, "") else "—"

        table.append([
            rank, f"{timestamp:.2f}s", frame_index, f"{final_score:.3f}",
            metric("aesthetic_normalized"), metric("sharpness_score"),
            metric("visibility_score"), metric("composition_score"),
        ])
    return gallery, table


def _score_table_html(rows: list[list[object]]) -> str:
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    if not body:
        body = f'<tr><td colspan="{len(SCORE_HEADERS)}">Scores will appear after analysis.</td></tr>'
    headers = "".join(f"<th>{html.escape(header)}</th>" for header in SCORE_HEADERS)
    return (
        '<p class="vmf-score-title">Scores</p>'
        '<table class="vmf-score-table"><thead><tr>'
        f'{headers}</tr></thead><tbody>{body}</tbody></table>'
    )


def analyze_video(
    source_mode,
    uploaded_video,
    input_video,
    yolo_device,
    aesthetic_device,
    confidence,
    sample_seconds,
    top_k,
    min_gap,
    max_seconds,
    ranking_preset,
    progress=gr.Progress(),
):
    """Run the existing pipeline and format its outputs for Gradio."""
    try:
        if (yolo_device == "0" or aesthetic_device == "cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA was selected, but PyTorch cannot access an NVIDIA GPU.")
        confidence = float(confidence)
        sample_seconds = float(sample_seconds)
        min_gap = float(min_gap)
        top_k = int(top_k)
        max_seconds = _optional_positive(max_seconds, "Max seconds")
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")
        if not math.isfinite(sample_seconds) or sample_seconds <= 0:
            raise ValueError("Sample interval must be positive.")
        if not math.isfinite(min_gap) or min_gap < 0:
            raise ValueError("Minimum gap must be nonnegative.")
        if top_k < 1:
            raise ValueError("Top K must be at least 1.")
        if ranking_preset not in PRESET_LABELS:
            raise ValueError("Choose a valid ranking preference.")

        progress(0.04, desc="Preparing video")
        source = _resolve_source(source_mode, uploaded_video, input_video)
        args = argparse.Namespace(
            input=str(source),
            output=None,
            model="yolo26n.pt",
            device=yolo_device,
            aesthetic_device=aesthetic_device,
            confidence=confidence,
            target_id=None,
            sample_seconds=sample_seconds,
            top_k=top_k,
            min_gap=min_gap,
            max_seconds=max_seconds,
            ranking_preset=ranking_preset,
        )
        progress(0.12, desc="Tracking person and scoring candidate frames")
        output = Path(run(args)).resolve()
        progress(0.96, desc="Loading selected moments")
        gallery, table = _load_results(output)
        status = (
            '<div class="vmf-success"><strong>Analysis complete.</strong> '
            f'Selected {len(gallery)} moment(s) from <code>{html.escape(source.name)}</code> '
            f'using <strong>{html.escape(PRESET_LABELS[ranking_preset])}</strong>.</div>'
        )
        progress(1.0, desc="Complete")
        return status, gallery, _score_table_html(table), str(output)
    except Exception as error:
        raise gr.Error(f"Analysis failed: {error}") from error


def build_theme():
    return gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#ffffff",
        body_background_fill_dark="#ffffff",
        body_text_color="#10213f",
        body_text_color_dark="#10213f",
        block_background_fill="#f7faff",
        block_background_fill_dark="#f7faff",
        block_border_color="#dbe4f0",
        block_border_color_dark="#dbe4f0",
        input_background_fill="#ffffff",
        input_background_fill_dark="#ffffff",
        input_border_color="#cbd8e8",
        input_border_color_dark="#cbd8e8",
        input_placeholder_color="#94a3b8",
        input_placeholder_color_dark="#94a3b8",
    )


def build_app() -> gr.Blocks:
    input_choices = list_input_videos()
    cuda_available = torch.cuda.is_available()
    default_yolo_device = "0" if cuda_available else "cpu"
    default_aesthetic_device = "cuda" if cuda_available else "cpu"

    with gr.Blocks(title="Video Moment Finder", fill_width=True) as app:
        source_mode = gr.State("upload")
        gr.Markdown(
            "# Video Moment Finder\nUpload or select a video, then find its best-looking moments.",
            elem_id="vmf-header",
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=4, min_width=340, elem_id="control-panel"):
                with gr.Tabs():
                    with gr.Tab("Upload video") as upload_tab:
                        uploaded_video = gr.Video(
                            label="Video file",
                            sources=["upload"],
                            show_label=False,
                        )
                    with gr.Tab("Input folder") as folder_tab:
                        input_video = gr.Dropdown(
                            choices=input_choices,
                            value=input_choices[0] if input_choices else None,
                            label="Video in input/",
                            info="Choose an existing video from the project's input folder.",
                        )
                        refresh_button = gr.Button("Refresh input folder", variant="secondary")

                ranking_preset = gr.Radio(
                    choices=PRESET_CHOICES,
                    value="balanced",
                    label="Ranking preference",
                    info="Choose what the ranking should prioritize; no numeric tuning required.",
                    elem_id="ranking-preset",
                )

                gr.Markdown("### Analysis settings")
                with gr.Row():
                    yolo_device = gr.Dropdown(
                        choices=[("GPU · CUDA device 0", "0"), ("CPU", "cpu")],
                        value=default_yolo_device,
                        label="YOLO device",
                    )
                    aesthetic_device = gr.Dropdown(
                        choices=[("GPU · CUDA", "cuda"), ("CPU", "cpu")],
                        value=default_aesthetic_device,
                        label="Aesthetic device",
                    )
                confidence = gr.Slider(0.05, 1.0, value=0.25, step=0.05, label="Detection confidence")
                with gr.Row():
                    sample_seconds = gr.Number(value=0.5, minimum=0.05, label="Sample interval (seconds)")
                    top_k = gr.Number(value=5, minimum=1, precision=0, label="Top K")
                with gr.Row():
                    min_gap = gr.Number(value=1.0, minimum=0, label="Minimum gap (seconds)")
                    max_seconds = gr.Textbox(
                        value="",
                        label="Max seconds (optional)",
                        placeholder="Full video",
                    )
                analyze_button = gr.Button("Analyze video", variant="primary", elem_id="analyze-button")

            with gr.Column(scale=8, min_width=560, elem_id="results-panel"):
                gr.Markdown("## Selected moments", elem_id="results-heading")
                status = gr.HTML(
                    '<div class="vmf-empty"><strong>Ready for analysis.</strong><br>'
                    'Upload a video or choose one from the input folder.</div>'
                )
                gallery = gr.Gallery(
                    label="Best frames",
                    show_label=False,
                    columns=5,
                    rows=1,
                    height="auto",
                    object_fit="cover",
                    interactive=False,
                    elem_id="moments-gallery",
                )
                scores = gr.HTML(
                    _score_table_html([]),
                    elem_id="moments-table",
                )
                output_folder = gr.Textbox(label="Output folder", interactive=False)

        upload_tab.select(lambda: "upload", outputs=source_mode, queue=False)
        folder_tab.select(lambda: "folder", outputs=source_mode, queue=False)
        refresh_button.click(refresh_input_choices, outputs=input_video, queue=False)
        analyze_button.click(
            analyze_video,
            inputs=[
                source_mode,
                uploaded_video,
                input_video,
                yolo_device,
                aesthetic_device,
                confidence,
                sample_seconds,
                top_k,
                min_gap,
                max_seconds,
                ranking_preset,
            ],
            outputs=[status, gallery, scores, output_folder],
        )

    return app


if __name__ == "__main__":
    build_app().queue(default_concurrency_limit=1).launch(theme=build_theme(), css=APP_CSS)
