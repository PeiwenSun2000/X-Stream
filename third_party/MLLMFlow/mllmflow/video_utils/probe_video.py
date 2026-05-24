import os
import subprocess
import json
import time
from pathlib import Path

def probe_video(video_path: str) -> dict:
    """Quickly probe complete video metadata without decoding frames, returning a dictionary with these fields:
        - duration: Video duration (seconds)
        - file_size: File size (bytes)
        - total_bitrate: Total bitrate (bps)
        - video: Video stream information (dictionary containing codec, width, height, fps, bitrate, etc.)
        - audio: Audio stream information (dictionary containing codec, sample_rate, channels, etc.; None if there is no audio)

        Only reads file header metadata and typically finishes within 50 ms.
        """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"FileNotFile {video_path}")
        return None

    attempts = max(1, int(os.environ.get("FLOW_VIDEO_PROBE_RETRIES", "3")))
    last_error = None
    for attempt in range(attempts):
        try:
            # Use -show_entries to fetch only the required fields, avoiding redundant parsing and improving speed
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

            # Initialize the result structure
            info = {
                'duration': float(data['format'].get('duration', 0)),
                'file_size': int(data['format'].get('size', video_path.stat().st_size)),
                'total_bitrate': int(data['format'].get('bit_rate', 0)),
                'video': None,
                'audio': None
            }

            # Iterate over streams and extract the first video stream and first audio stream
            for stream in data.get('streams', []):
                if stream['codec_type'] == 'video' and info['video'] is None:
                    # Parse frame rate (r_frame_rate format such as "30/1")
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

            # If the video bitrate is missing, estimate it from total bitrate by subtracting the audio bitrate
            if info['video'] and info['video']['bitrate'] == 0:
                audio_bitrate = info['audio']['bitrate'] if info['audio'] else 128000
                estimated_video_bitrate = max(info['total_bitrate'] - audio_bitrate, 1000000)
                info['video']['bitrate'] = estimated_video_bitrate

            return info

        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError, OSError) as e:
            last_error = e
            if attempt + 1 < attempts:
                time.sleep(min(2.0, 0.5 * (attempt + 1)))
                continue
    print(f"Failed to probe video {video_path}: {last_error}")
    return None
