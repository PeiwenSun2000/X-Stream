
def time_to_seconds(t: str) -> int:
    parts = t.split(':')
    if len(parts) == 2:
        m, s = map(int, parts)
        return m * 60 + s
    elif len(parts) == 3:
        h, m, s = map(int, parts)
        return h * 3600 + m * 60 + s
    else:
        raise ValueError(f"Invalid time format: {t}")

def seconds_to_time(sec: int) -> str:
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h <= 0:
        return f"{m:02d}:{s:02d}"
    else:
        return f"{h:02d}:{m:02d}:{s:02d}"

def generate_time_range(start_str: str, end_str: str, delta=1):
    start_sec = time_to_seconds(start_str)
    end_sec = time_to_seconds(end_str)
    return [seconds_to_time(sec) for sec in range(start_sec, end_sec + 1, delta)]

def format_time(t: str):
    if ':' not in str(t):
        return seconds_to_time(int(t))
    return seconds_to_time(time_to_seconds(t))

def delta_time(start_time: str, end_time: str) -> int:
    return time_to_seconds(start_time) - time_to_seconds(end_time)
