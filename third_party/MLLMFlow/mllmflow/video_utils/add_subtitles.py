import json
import subprocess
import os
import tempfile

def add_subtitles_to_video_ass(input_video, subtitles, output_video, font_path="fonts/NotoSansCJKsc-Regular.otf"):
    """
    Add subtitles using the .ass subtitle format to bypass drawtext length limits.
    Supports color/size/align (top/middle/bottom).
    """
    if isinstance(subtitles, str):
        with open(subtitles, encoding='utf-8') as f:
            subs = json.load(f)
    else:
        subs = subtitles

    # Validate inputs
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Video does not exist: {input_video}")
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font does not exist: {font_path}")
    font_name = os.path.splitext(os.path.basename(font_path))[0]

    # Map align to ASS alignment (1=bottom-left, 2=bottom-center, 3=bottom-right, 7=top-left, 8=top-center, 9=top-right)
    def get_alignment(align):
        return {"top": 8, "middle": 8, "bottom": 2}[align]

    # Create .ass content
    ass_lines = [
        "[Script Info]",
        "Title: Auto-generated Subtitles",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.601",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
    ]

    # Define a base style (color/size are dynamically overridden per subtitle item, but ASS does not support per-line font size)
    # so use a fixed size, or set it dynamically with \fs (recommended)
    ass_lines.append(
        f"Style: Default,{font_name},48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1"
    )
    ass_lines.append("")
    ass_lines.append("[Events]")
    ass_lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")

    for item in subs:
        text = item["text"].replace("\n", "\\N")  # ASS newline marker
        start = item["start"]
        end = item["end"]
        color = item.get("color", "white")
        size = item.get("size", 48)
        align = item.get("align", "middle")

        # Convert colors: ASS uses &HBBGGRR (with alpha as &HAABBGGRR, but usually &H00BBGGRR)
        color_map = {
            "white": "&H00FFFFFF",
            "yellow": "&H0000FFFF",
            "red": "&H000000FF"
        }
        ass_color = color_map.get(color, "&H00FFFFFF")

        # Time format: H:MM:SS.cc (centiseconds, not milliseconds)
        def fmt_time(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = int(t % 60)
            cs = int(round((t - int(t)) * 100))  # centiseconds
            return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

        start_str = fmt_time(start)
        end_str = fmt_time(end)

        # Dynamic settings: font size \fs, color \c, alignment \an
        alignment = get_alignment(align)
        event_text = f"{{\\an{alignment}\\fs{size}\\c{ass_color}}}{text}"
        ass_lines.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{event_text}")

    # Write the temporary .ass file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ass', delete=False, encoding='utf-8') as f:
        f.write("\n".join(ass_lines))
        ass_file = f.name

    try:
        cmd = [
            os.environ.get("FFMPEG_BIN", "ffmpeg"),
            "-y", "-i", input_video,
            "-vf", f"subtitles={ass_file}:fontsdir={os.path.dirname(font_path)}",
            "-c:a", "copy",
            output_video
        ]
        subprocess.run(cmd, check=True)
    finally:
        os.unlink(ass_file)  # Clean up the temporary file

# Generate high-frequency timestamps (0.1-second intervals, now safe!)
def add_timestamp_to_video(input_video, output_video):
    from video_utils.probe_video import probe_video
    duration = float(probe_video(input_video).get("duration", 1.0))
    subtitles = []
    step = 0.1
    num_steps = int(duration / step) + 1
    for i in range(num_steps):
        start = i * step
        end = min((i + 1) * step, duration)
        if start >= duration:
            break
        subtitles.append({
            "text": f"timestamp: {start:.2f}s",
            "start": start,
            "end": end,
            "color": "red",
            "size": 24,
            "align": "top"
        })
    add_subtitles_to_video_ass(input_video, subtitles, output_video)
    
if __name__ == "__main__":
    add_timestamp_to_video(
        input_video="./assets/tmp/video_video_wo_t_0240_0330_fps_1.0.mp4",
        output_video="./assets/tmp/video_video_w_t_0240_0330_fps_1.0.mp4",
    )