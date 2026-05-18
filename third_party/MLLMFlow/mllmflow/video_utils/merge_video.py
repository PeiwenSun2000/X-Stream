#!/usr/bin/env python3
import os
import subprocess
import tempfile
from pathlib import Path


def merge_video(clip_list, output_path=None):
    """
    极速拼接多个视频片段（要求编码参数一致，如你的 trim_video 输出）。

    Args:
        clip_list (List[str]): 视频片段路径列表，按时间顺序
        output_path (str, optional): 输出路径。若为 None，则自动生成（如 merged_output.mp4）

    Returns:
        str: 输出视频的绝对路径
    """
    if not clip_list:
        raise ValueError("clip_list is empty")

    # 检查所有文件存在
    for p in clip_list:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Clip not found: {p}")

    # 自动生成输出路径
    if output_path is None:
        first = Path(clip_list[0])
        output_path = first.parent / f"{first.stem}_merged.mp4"

    output_path = os.path.abspath(output_path)

    # 使用临时文件写入 filelist
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for clip in clip_list:
            # ffprobe 要求路径转义单引号（但通常不需要），这里用绝对路径+安全模式
            f.write(f"file '{os.path.abspath(clip)}'\n")
        list_file = f.name

    try:
        # 调用 ffmpeg 快速拼接（不重新编码！）
        subprocess.run([
            "ffmpeg",
            "-y",                    # 覆盖输出
            "-f", "concat",         # 使用 concat demuxer
            "-safe", "0",           # 允许绝对路径
            "-i", list_file,
            "-c", "copy",           # ⚡ 关键：流拷贝，不转码！
            "-movflags", "+faststart",  # 优化网络播放
            output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        os.unlink(list_file)  # 清理临时文件

    return output_path


if __name__ == "__main__":
    # 简单测试
    import glob
    clips = sorted(glob.glob("tmp/*.mp4"))
    print(clips)
    merged_path = merge_video(clips)
    print(f"Merged video saved to: {merged_path}")
