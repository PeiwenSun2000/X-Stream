#!/usr/bin/env python3
import os
import subprocess
import tempfile
from pathlib import Path


def merge_video(clip_list, output_path=None):
    """
    Rapidly concatenate multiple video clips (requires matching encoding parameters, such as trim_video output).

    Args:
        clip_list (List[str]): List of video clip paths in chronological order
        output_path (str, optional): Output path. If None, it is generated automatically (for example, merged_output.mp4)

    Returns:
        str: Absolute path of the output video
    """
    if not clip_list:
        raise ValueError("clip_list is empty")

    # Check that all files exist
    for p in clip_list:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Clip not found: {p}")

    # Automatically generate the output path
    if output_path is None:
        first = Path(clip_list[0])
        output_path = first.parent / f"{first.stem}_merged.mp4"

    output_path = os.path.abspath(output_path)

    # Write the file list to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for clip in clip_list:
            # ffprobe requires single quotes in paths to be escaped (usually unnecessary); use absolute paths plus safe mode here
            f.write(f"file '{os.path.abspath(clip)}'\n")
        list_file = f.name

    try:
        # Call ffmpeg for fast concatenation without re-encoding
        subprocess.run([
            "ffmpeg",
            "-y",                    # Overwrite output
            "-f", "concat",         # Use the concat demuxer
            "-safe", "0",           # Allow absolute paths
            "-i", list_file,
            "-c", "copy",           # ⚡ Important: stream copy, no transcoding!
            "-movflags", "+faststart",  # Optimize for network playback
            output_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        os.unlink(list_file)  # Clean up the temporary file

    return output_path


if __name__ == "__main__":
    # Simple test
    import glob
    clips = sorted(glob.glob("tmp/*.mp4"))
    print(clips)
    merged_path = merge_video(clips)
    print(f"Merged video saved to: {merged_path}")
