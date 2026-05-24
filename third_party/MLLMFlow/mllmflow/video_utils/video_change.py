"""Estimate video clip change magnitude, used by code mode to select the stream with larger changes."""
from __future__ import annotations

import warnings

import numpy as np
from moviepy import VideoFileClip


def video_change_score(clip_path: str, num_samples: int = 8) -> float:
    """
    Sample several frames from a video clip and compute the sum of pixel differences between adjacent frames as the change-magnitude scalar.
    Larger changes produce larger return values, used for comparison with another stream.

    Args:
        clip_path: Video clip path (usually a short clip after trimming)
        num_samples: Desired number of sampled frames (automatically reduced for short clips to avoid MoviePy warnings caused by reading past the file tail)

    Returns:
        Non-negative float; returns 0.0 if it cannot be computed
    """
    try:
        with warnings.catch_warnings():
            # For short clips (for example, 1s/2fps has only 2 frames), MoviePy can trigger a "0 bytes read" warning when reading the last frame, so suppress that warning
            warnings.simplefilter("ignore", UserWarning)
            with VideoFileClip(clip_path) as clip:
                duration = clip.duration
                if duration <= 0 or num_samples < 2:
                    return 0.0
                # Short clips may only have 2-3 frames, so keep the sample count around duration*2 and ensure timestamps stay strictly inside the clip to avoid out-of-range reads
                n = min(num_samples, max(2, int(duration * 2.5)))
                n = max(2, n)
                t_end = max(0.0, duration - 0.05)
                if t_end <= 0:
                    return 0.0
                times = np.linspace(0, t_end, num=n)
                frames = [clip.get_frame(float(t)) for t in times]
    except Exception:
        return 0.0

    if len(frames) < 2:
        return 0.0

    total_diff = 0.0
    for i in range(1, len(frames)):
        # Convert to float before subtraction to avoid overflow
        a = np.asarray(frames[i - 1], dtype=np.float64)
        b = np.asarray(frames[i], dtype=np.float64)
        total_diff += np.abs(a - b).sum()
    return float(total_diff)
