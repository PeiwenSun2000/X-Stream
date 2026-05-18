import subprocess
import os

def extract_frame_ffmpeg(video_path: str, output_path: str, time_sec: float) -> bool:
    """
    从视频第 time_sec 秒提取一帧，保存为 output_path。
    成功返回 True，失败返回 False。
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
        # 检查返回码，不抛出异常
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False  # 关键：不自动抛异常
        )
        return result.returncode == 0
    except Exception:
        return False

if __name__=="__main__":
    # 使用示例
    print(extract_frame_ffmpeg("land.mp4", "land.jpg", 3.0))  # 提取第 10.5 秒
