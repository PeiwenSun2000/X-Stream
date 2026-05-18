import os
import subprocess
import json
from pathlib import Path

def probe_video(video_path: str) -> dict:
    """快速探测视频文件的完整元数据（不解码帧），返回包含以下字段的字典：
        - duration: 视频时长（秒）
        - file_size: 文件大小（字节）
        - total_bitrate: 总码率（bps）
        - video: 视频流信息（字典，含 codec、width、height、fps、bitrate 等）
        - audio: 音频流信息（字典，含 codec、sample_rate、channels 等，若无音频则为 None）

        仅读取文件头元数据，执行通常在 50ms 以内。
        """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"FileNotFile {video_path}")
        return None

    try:
        # 使用 -show_entries 精确获取所需字段，避免解析冗余信息，提升速度
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries",
            "format=duration,size,bit_rate:"
            "stream=index,codec_name,codec_type,profile,level,bit_rate,"
            "width,height,r_frame_rate,avg_frame_rate,pix_fmt,nb_frames,"
            "sample_rate,channels,channel_layout",
            "-of", "json",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        # 初始化结果结构
        info = {
            'duration': float(data['format'].get('duration', 0)),
            'file_size': int(data['format'].get('size', video_path.stat().st_size)),
            'total_bitrate': int(data['format'].get('bit_rate', 0)),
            'video': None,
            'audio': None
        }

        # 遍历流，提取第一个视频流和第一个音频流
        for stream in data.get('streams', []):
            if stream['codec_type'] == 'video' and info['video'] is None:
                # 解析帧率（r_frame_rate 格式如 "30/1"）
                fps_str = stream.get('r_frame_rate', '30/1')
                try:
                    num, den = map(float, fps_str.split('/'))
                    fps = num / den if den != 0 else 30.0
                except (ValueError, ZeroDivisionError):
                    fps = 30.0

                info['video'] = {
                    'index': stream.get('index', 0),
                    'codec': stream.get('codec_name', 'h264'),
                    'profile': stream.get('profile', 'Main'),
                    'level': stream.get('level', 0),
                    'width': int(stream.get('width', 0)),
                    'height': int(stream.get('height', 0)),
                    'fps': fps,
                    'total_frames': int(stream.get('nb_frames', fps * info['duration'])),
                    'pix_fmt': stream.get('pix_fmt', 'yuv420p'),
                    'bitrate': int(stream.get('bit_rate', 0)) if stream.get('bit_rate') else 0
                }

            elif stream['codec_type'] == 'audio' and info['audio'] is None:
                info['audio'] = {
                    'index': stream.get('index', 1),
                    'codec': stream.get('codec_name', 'aac'),
                    'sample_rate': int(stream.get('sample_rate', 48000)) if stream.get('sample_rate') else 48000,
                    'channels': int(stream.get('channels', 2)) if stream.get('channels') else 2,
                    'channel_layout': stream.get('channel_layout', 'stereo'),
                    'bitrate': int(stream.get('bit_rate', 128000)) if stream.get('bit_rate') else 128000
                }

        # 如果视频码率缺失，尝试从总码率估算（减去音频码率）
        if info['video'] and info['video']['bitrate'] == 0:
            audio_bitrate = info['audio']['bitrate'] if info['audio'] else 128000
            estimated_video_bitrate = max(info['total_bitrate'] - audio_bitrate, 1000000)
            info['video']['bitrate'] = estimated_video_bitrate

        return info

    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError, OSError) as e:
        print(f"Failed to probe video {video_path}: {e}")
        return None
