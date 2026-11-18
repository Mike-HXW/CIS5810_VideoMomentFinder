"""Save reference frames alongside the main ranking results."""

import math
import re
import shutil
import subprocess

import cv2


def ffmpeg_thumbnail(source, destination, frame_count):
    """Choose a representative frame from regularly spaced samples of the full video."""
    from imageio_ffmpeg import count_frames_and_secs, get_ffmpeg_exe

    executable = get_ffmpeg_exe()
    if not math.isfinite(frame_count) or frame_count < 1:
        frame_count, _ = count_frames_and_secs(str(source))
    batch_size = 100
    stride = max(1, math.ceil(frame_count / batch_size))
    command = [executable, '-nostdin', '-hide_banner', '-nostats', '-i', str(source),
               '-map', '0:V:0', '-fps_mode', 'passthrough']

    # Keep thumbnails small while scoring. Preserve the decoded frame index in
    # PTS so the chosen frame can then be saved at its original resolution.
    while True:
        filters = (f'setpts=N,select=not(mod(n\\,{stride})),'
                   'scale=320:320:force_original_aspect_ratio=decrease,'
                   f'thumbnail=n={batch_size}:log=quiet,showinfo')
        result = subprocess.run(
            [*command, '-vf', filters, '-f', 'null', '-'],
            capture_output=True, check=True,
        )
        indices = re.findall(rb'\bn:\s*\d+\s+pts:\s*(\d+)\s+pts_time:', result.stderr)
        if not indices:
            raise RuntimeError('No representative frame was returned.')
        if len(indices) == 1:
            frame_index = int(indices[0])
            break
        # Some containers underreport frame counts. Increase spacing if the
        # full scan produced multiple batches so the last batch is not ignored.
        stride *= len(indices)

    result = subprocess.run(
        [*command, '-loglevel', 'error', '-vf', f'select=eq(n\\,{frame_index})',
         '-frames:v', '1', '-c:v', 'png', '-f', 'image2pipe', '-'],
        capture_output=True, check=True,
    )
    if not result.stdout:
        raise RuntimeError(f'Could not extract representative frame {frame_index}.')
    destination.write_bytes(result.stdout)
    return {'frame_index': frame_index, 'sample_stride_frames': stride,
            'batch_size': batch_size, 'scope': 'whole_video'}


def save_baselines(source, output, candidates, frame_count, frames=None):
    """Reuse the existing aesthetic scores and export one image per method.

    ``frames`` maps frame index -> full-quality decoded frame; when the top frame is
    available it is written as PNG (candidate images themselves are JPEG previews).
    """
    aesthetic_path = 'baselines/aesthetic_only/top_1.png'
    ffmpeg_path = 'baselines/ffmpeg_thumbnail/top_1.png'
    (output / aesthetic_path).parent.mkdir(parents=True)
    (output / ffmpeg_path).parent.mkdir(parents=True)

    # Use the raw score: percentile clipping can tie different aesthetic scores.
    top = min(candidates, key=lambda candidate: (-candidate['aesthetic_score'], candidate['frame_index']))
    frame = (frames or {}).get(top['frame_index'])
    if frame is not None:
        if not cv2.imwrite(str(output / aesthetic_path), frame):
            raise OSError(f'Cannot save {aesthetic_path}')
    else:
        shutil.copy2(output / top['image'], output / aesthetic_path)
    results = {'aesthetic_only': {
        'image': aesthetic_path, 'frame_index': top['frame_index'],
        'timestamp_seconds': top['timestamp_seconds'], 'aesthetic_score': top['aesthetic_score'],
        'scope': 'tracked_candidates' if top.get('track_id') is not None else 'scene_candidates',
    }}

    try:
        thumbnail = ffmpeg_thumbnail(source, output / ffmpeg_path, frame_count)
    except (ImportError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        if isinstance(error, subprocess.CalledProcessError):
            lines = error.stderr.decode(errors='replace').strip().splitlines()
            detail = lines[-1] if lines else str(error)
        else:
            detail = str(error)
        print(f'Warning: FFmpeg baseline could not be saved: {detail}', flush=True)
        results['ffmpeg_thumbnail'] = {'status': 'error', 'error': detail}
    else:
        results['ffmpeg_thumbnail'] = {'image': ffmpeg_path, **thumbnail}
    return results
