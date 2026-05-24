import subprocess
import os

def extract_frame_ffmpeg(video_path: str, output_path: str, time_sec: float) -> bool:
    """
    Extract one frame at time_sec seconds from the video and save it to output_path.
    Return True on success and False on failure.
    """
    if not os.path.exists(video_path):
        return False

    cmd = [
        "ffmpeg",
        "-ss", str(time_sec),
        "-i", video_path,
        "-vframes", "1",
        "-q:v", "1",
        "-y",
        output_path
    ]
    try:
        # Check the return code without raising exceptions
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False  # Important: do not raise exceptions automatically
        )
        return result.returncode == 0
    except Exception:
        return False

if __name__=="__main__":
    # Usage example
    print(extract_frame_ffmpeg("land.mp4", "land.jpg", 3.0))  # Extract frame at 10.5 seconds
