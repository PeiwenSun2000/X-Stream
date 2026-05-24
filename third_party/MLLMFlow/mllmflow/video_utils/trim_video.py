import os
import shutil
import tempfile
from pathlib import Path
# from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy import VideoFileClip
from mllmflow.time_utils.time_utils import time_to_seconds
from mllmflow.video_utils.probe_video import probe_video

def trim_video(
    video_path: str,
    trim_path: str,
    start_time: str,
    end_time: str,
    temp_dir: str = "moviepy_tmp",
    fps: float = None, # fps sampling rate
) -> dict:
    """
    Trim video to specified time range, using original video/audio bitrate to preserve quality.
    Output file size will naturally be smaller than original (since it's a clip).
    """

    verbose = os.environ.get("FLOW_VERBOSE_CACHE", "").lower() in ("1", "true", "yes")
    if verbose:
        print(video_path, trim_path)

    # Probe original video
    orig_info = probe_video(video_path)
    if orig_info is None:
        raise ValueError(f"Failed to probe original video: {video_path}")

    # Create temp base dir
    base_temp = Path(temp_dir)
    base_temp.mkdir(parents=True, exist_ok=True)

    start = time_to_seconds(start_time)
    end = time_to_seconds(end_time)

    with VideoFileClip(video_path) as video:
        duration = float(video.duration or 0.0)
        if duration <= 0.0:
            raise ValueError(f"Invalid video duration ({duration}) for: {video_path}")

        # Clamp to [0, duration] and avoid floating-point overshoot at the tail.
        start = max(0.0, min(float(start), duration))
        end = max(0.0, min(float(end), duration))

        # Prefer a 1s window when possible, but never exceed real duration.
        if start >= end or (end - start) < 1.0:
            if duration >= 1.0:
                end = min(duration, max(end, start + 1.0))
                start = max(0.0, end - 1.0)
            else:
                start = 0.0
                end = duration

        # Keep end strictly inside duration to satisfy MoviePy checks.
        end = min(end, max(duration - 1e-6, 0.0))
        if start >= end:
            start = max(0.0, end - min(0.1, max(duration, 0.1) / 2))
        if start >= end:
            raise ValueError(
                f"Cannot trim video with non-positive window: start={start}, end={end}, duration={duration}, video={video_path}"
            )

        clip = video.subclipped(start, end)
        has_audio = video.audio is not None

        # Estimate target duration
        duration = max(clip.duration, 0.1)

        # Determine video and audio bitrates from original
        video_bitrate = 0
        audio_bitrate = 128000  # default fallback

        if orig_info.get('audio') and orig_info['audio'].get('bitrate'):
            audio_bitrate = orig_info['audio']['bitrate']
        elif has_audio:
            audio_bitrate = 128000

        if orig_info.get('video') and orig_info['video'].get('bitrate'):
            video_bitrate = orig_info['video']['bitrate']
        elif orig_info.get('total_bitrate'):
            # Fallback: assume audio is 128kbps, rest is video
            video_bitrate = max(orig_info['total_bitrate'] - audio_bitrate, 1)

        # Convert to kbps for MoviePy
        video_bitrate_kbps = max(1, int(video_bitrate / 1000))
        audio_bitrate_kbps = max(1, int(audio_bitrate / 1000)) if has_audio else None

        # Write with original-like quality
        with tempfile.TemporaryDirectory(dir=base_temp) as tmp_dir:
            tmp_out = Path(tmp_dir) / "out.mp4"
            try:
                clip.write_videofile(
                    str(tmp_out),
                    codec="libx264",
                    audio_codec="aac" if has_audio else None,
                    bitrate=f"{video_bitrate_kbps}k",
                    audio_bitrate=f"{audio_bitrate_kbps}k" if has_audio else None,
                    preset="medium",  # balance speed/quality
                    fps=fps, ## increase fps sampling rate
                    logger=None,
                    temp_audiofile=str(Path(tmp_dir) / "aud.m4a"),
                    remove_temp=True,
                    ffmpeg_params=["-movflags", "+faststart", "-avoid_negative_ts", "make_zero", "-force_key_frames", "0"]
                )
                if tmp_out.exists():
                    shutil.move(str(tmp_out), trim_path)

            except Exception as e:
                print(f"⚠️ Encoding failed: {e}", video_path, trim_path)
                raise
                return None

    # Final probe and return
    info = probe_video(trim_path)
    if info is not None:
        info["size_bytes"] = info["file_size"]
        info["size_mb"] = info["file_size"] / (1024 * 1024)
        if verbose:
            print(f"OK {trim_path} | video_bitrate={video_bitrate_kbps}k | "
                  f"audio_bitrate={audio_bitrate_kbps}k | {info['size_mb']:.2f} MB")
    return info
