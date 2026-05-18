"""视频片段变化量估计，用于 code 模式选择变化较大的 stream。"""
from __future__ import annotations

import warnings

import numpy as np
from moviepy import VideoFileClip


def video_change_score(clip_path: str, num_samples: int = 8) -> float:
    """
    对视频片段采样若干帧，计算相邻帧之间的像素差异之和，作为“变化量”标量。
    变化越大返回值越大，用于与另一路 stream 比较。

    Args:
        clip_path: 视频片段路径（通常为 trim 后的短 clip）
        num_samples: 期望采样帧数（短片段会自动减少，避免读到文件尾触发 MoviePy 警告）

    Returns:
        非负浮点数，若无法计算则返回 0.0
    """
    try:
        with warnings.catch_warnings():
            # 短 clip（如 1s/2fps 仅 2 帧）时 MoviePy 读末尾帧易触发 "0 bytes read"，抑制该警告
            warnings.simplefilter("ignore", UserWarning)
            with VideoFileClip(clip_path) as clip:
                duration = clip.duration
                if duration <= 0 or num_samples < 2:
                    return 0.0
                # 短片段可能只有 2～3 帧，采样数不超过约 (duration*2)，且时间严格落在片段内避免读越界
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
        # 转为 float 再差，避免溢出
        a = np.asarray(frames[i - 1], dtype=np.float64)
        b = np.asarray(frames[i], dtype=np.float64)
        total_diff += np.abs(a - b).sum()
    return float(total_diff)
