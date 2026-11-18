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
from subject_preview import scan_subjects


VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
PRESET_CHOICES = [
    ("Balanced", "balanced"),
    ("Clear subject", "clear-subject"),
    ("Subject fully shown", "subject-fully-shown"),
    ("Best-looking scene", "best-looking-scene"),
]
PRESET_LABELS = {value: label for label, value in PRESET_CHOICES}
SCAN_CACHE_SIZE = 2  # Enough for one upload plus one input-folder video per session.
SCENE_LABEL = 'Entire scene selected'
# Uploaded temp file -> its copy in input/, so each upload is copied only once.
_UPLOAD_COPIES: dict[str, Path] = {}
SCORE_HEADERS = [
    "Rank", "Timestamp", "Frame", "Final", "Aesthetic",
    "Sharpness", "Visibility", "Composition",
]


APP_CSS = """
.gradio-container { max-width: 1500px !important; margin: 0 auto !important; }
#control-panel, #results-panel { border: 1px solid #dbe4f0; border-radius: 14px; padding: 16px; }
.vmf-empty { border: 1px dashed #c8d5e6; border-radius: 12px; color: #64748b; padding: 44px 24px; text-align: center; }
.vmf-success { background: #eaf9f0; border: 1px solid #bce8cb; border-radius: 10px; color: #17653a; padding: 12px 14px; }
.vmf-score-title { font-size: 0.9rem; margin: 0 0 8px; }
.vmf-score-table { border-collapse: collapse; width: 100%; font-size: 0.88rem; }
.vmf-score-table th, .vmf-score-table td { border-bottom: 1px solid #dbe4f0; padding: 8px 10px; text-align: left; }
.vmf-score-table th { background: #eef4fb; }
@media (max-width: 900px) { .vmf-score-table { display: block; overflow-x: auto; white-space: nowrap; } }
"""
# Keep the page in light mode even when the OS uses dark mode, so no dark-mode overrides are needed.
FORCE_LIGHT_JS = "() => document.body.classList.remove('dark')"


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
    return gr.Dropdown(choices=choices, value=None)


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
        # One copy per upload, even if a stopped scan already copied it.
        copied = _UPLOAD_COPIES.get(str(uploaded_path))
        if copied is not None and copied.is_file():
            return copied
        destination = INPUT_DIR / _safe_uploaded_name(uploaded_path)
        shutil.copy2(uploaded_path, destination)
        _UPLOAD_COPIES[str(uploaded_path)] = destination.resolve()
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


def _source_key(source_mode, uploaded_video, input_video):
    """Identify the video currently chosen in the UI (before any copy into input/)."""
    if source_mode == 'upload':
        return ('upload', str(uploaded_video)) if uploaded_video else None
    return ('folder', str(input_video)) if input_video else None


def preview_subjects(source_mode, uploaded_video, input_video, yolo_device, confidence, cache):
    """Scan the chosen video for people (runs when the user clicks Find people)."""
    cache = dict(cache or {})
    empty = (None, None, [], [], SCENE_LABEL, gr.update(visible=False), cache)
    key = _source_key(source_mode, uploaded_video, input_video)
    if key is None:
        gr.Warning('Choose a video first.')
        return empty
    # Same video and same scan settings -> reuse the scan instead of copying and rescanning.
    cache_key = (*key, str(yolo_device), float(confidence))
    scan = cache.get(cache_key)
    if scan is None:
        try:
            source = _resolve_source(source_mode, uploaded_video, input_video)
            scan = scan_subjects(source, str(PROJECT_DIR / 'yolo26s.pt'), yolo_device, float(confidence))
        except Exception as error:
            # Clear the scan so Analyze can never fall back to the previous video.
            gr.Warning(f'Subject preview failed: {error}')
            return empty
        scan['origin'] = key
        cache[cache_key] = scan
        while len(cache) > SCAN_CACHE_SIZE:
            cache.pop(next(iter(cache)))
    recommended = [track for track in scan['tracks'] if track['recommended']][:5]
    shown_ids = {track['id'] for track in recommended}
    more = [track for track in scan['tracks'] if track['id'] not in shown_ids]
    scan['recommended_ids'] = [track['id'] for track in recommended]
    scan['more_ids'] = [track['id'] for track in more]
    def gallery_items(tracks):
        return [(track['preview'], f"Person {track['id']} · {track['seconds']:.1f}s") for track in tracks]
    return (scan, None, gallery_items(recommended), gallery_items(more), SCENE_LABEL,
            gr.update(visible=bool(more)), cache)


def reset_subjects():
    """A new video (or source tab) starts as Entire scene with no people listed."""
    return None, None, [], [], SCENE_LABEL, gr.update(visible=False)


def select_subject(scan, chosen, group, evt: gr.SelectData):
    """Clicking a person adds or removes them; several entries are treated as one subject
    (tracks are split at hard cuts, so one person can appear as several entries)."""
    person_id = scan[group][evt.index]
    chosen = [p for p in (chosen or []) if p != person_id] + ([] if person_id in (chosen or []) else [person_id])
    if not chosen:
        return None, SCENE_LABEL
    names = ' + '.join(f'Person {p}' for p in chosen)
    hint = ' (treated as one person)' if len(chosen) > 1 else ''
    return chosen, f'{names} selected{hint} · click again to remove'


def select_recommended(scan, chosen, evt: gr.SelectData):
    return select_subject(scan, chosen, 'recommended_ids', evt)


def select_more(scan, chosen, evt: gr.SelectData):
    return select_subject(scan, chosen, 'more_ids', evt)


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
    scan,
    selected_subject,
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

        current = _source_key(source_mode, uploaded_video, input_video)
        if current is None:
            raise ValueError('Choose a video before starting the analysis.')
        scene = selected_subject is None
        if scene:
            # Entire scene needs no person scan; reuse a finished one's copied source if present.
            scan_ok = scan is not None and scan.get('origin') == current
            source = Path(scan['source']) if scan_ok else _resolve_source(source_mode, uploaded_video, input_video)
            scan = None
        else:
            if scan is None or scan.get('origin') != current:
                raise ValueError('Click "Find people" for this video and choose a person first.')
            source = Path(scan['source'])
        args = argparse.Namespace(
            input=str(source),
            output=None,
            model="yolo26s.pt",
            device=yolo_device,
            aesthetic_device=aesthetic_device,
            confidence=confidence,
            target_id=None if scene else sorted(selected_subject),
            scene=scene,
            scan=scan,
            sample_seconds=sample_seconds,
            top_k=top_k,
            min_gap=min_gap,
            max_seconds=max_seconds,
            ranking_preset=ranking_preset,
        )
        output = Path(run(args)).resolve()
        gallery, table = _load_results(output)
        status = (
            '<div class="vmf-success"><strong>Analysis complete.</strong> '
            f'Selected {len(gallery)} moment(s) from <code>{html.escape(source.name)}</code> '
            f'using <strong>{html.escape("Entire scene" if selected_subject is None else PRESET_LABELS[ranking_preset])}</strong>.</div>'
        )
        return status, gallery, _score_table_html(table), str(output)
    except Exception as error:
        raise gr.Error(f"Analysis failed: {error}") from error


def build_theme():
    return gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_text_color="#10213f",
        block_background_fill="#f7faff",
        block_border_color="#dbe4f0",
        input_border_color="#cbd8e8",
    )


def build_app() -> gr.Blocks:
    input_choices = list_input_videos()
    cuda_available = torch.cuda.is_available()
    default_yolo_device = "0" if cuda_available else "cpu"
    default_aesthetic_device = "cuda" if cuda_available else "cpu"

    with gr.Blocks(title="Video Moment Finder", fill_width=True) as app:
        source_mode = gr.State("upload")
        scan_state = gr.State(None)
        scan_cache = gr.State({})
        selected_subject = gr.State(None)
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
                            value=None,
                            label="Video in input/",
                            info="Choose an existing video from the project's input folder.",
                        )
                        refresh_button = gr.Button("Refresh input folder", variant="secondary")

                gr.Markdown('### Choose a subject')
                gr.Markdown('Analyze the entire scene, or click Find people to choose a person.')
                with gr.Row():
                    entire_scene = gr.Button('Entire scene', variant='secondary')
                    find_people = gr.Button('Find people', variant='secondary')
                subject_label = gr.Markdown(SCENE_LABEL)
                subjects = gr.Gallery(label='Recommended people', columns=5, height='auto', object_fit='contain', allow_preview=False)
                with gr.Accordion('Show more', open=False, visible=False) as more_section:
                    more_subjects = gr.Gallery(label='More people', columns=5, height='auto', object_fit='contain', allow_preview=False)

                ranking_preset = gr.Radio(
                    choices=PRESET_CHOICES,
                    value="balanced",
                    label="Ranking preference",
                    info="Applies when a person is selected. Entire scene uses aesthetics and whole-frame sharpness.",
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
                    sample_seconds = gr.Number(value=0.333, minimum=0.05, label="Sample interval (seconds)")
                    top_k = gr.Number(value=5, minimum=1, precision=0, label="Top K")
                with gr.Row():
                    min_gap = gr.Number(value=0.333, minimum=0, label="Minimum gap (seconds)")
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
        preview_outputs = [scan_state, selected_subject, subjects, more_subjects, subject_label, more_section]
        # Choosing a video no longer scans; it just resets the subject choice to Entire scene.
        uploaded_video.change(reset_subjects, outputs=preview_outputs, queue=False)
        input_video.change(reset_subjects, outputs=preview_outputs, queue=False)
        upload_tab.select(reset_subjects, outputs=preview_outputs, queue=False)
        folder_tab.select(lambda: (*reset_subjects(), None), outputs=preview_outputs + [input_video], queue=False)
        # The scan reuses the per-session cache, so clicking again for the same video is instant.
        find_people.click(
            preview_subjects,
            inputs=[source_mode, uploaded_video, input_video, yolo_device, confidence, scan_cache],
            outputs=preview_outputs + [scan_cache],
        )
        subjects.select(select_recommended, inputs=[scan_state, selected_subject], outputs=[selected_subject, subject_label])
        more_subjects.select(select_more, inputs=[scan_state, selected_subject], outputs=[selected_subject, subject_label])
        # Keeps the people list so the user can switch back to a person.
        entire_scene.click(lambda: (None, SCENE_LABEL), outputs=[selected_subject, subject_label], queue=False)
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
                scan_state,
                selected_subject,
            ],
            outputs=[status, gallery, scores, output_folder],
        )

    return app


if __name__ == "__main__":
    build_app().queue(default_concurrency_limit=1).launch(theme=build_theme(), css=APP_CSS, js=FORCE_LIGHT_JS)
