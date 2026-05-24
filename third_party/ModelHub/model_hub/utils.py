import os
import time
import json
import base64
from pathlib import Path
from json_repair import repair_json

def to_base64(path: str, max_size_bytes: int = 100 * 1024 * 1024) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if os.path.getsize(path) > max_size_bytes:
        raise ValueError(f"File size exceeds limit ({max_size_bytes // (1024*1024)} MB).")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def load_json(content):
    if isinstance(content, dict):
        return content
    try:
        return json.loads(repair_json(content))
    except:
        return {"content": content}

LOG_DIR = Path("logs")
MAX_FILES_PER_DIR = 50
_STATE_FILE = LOG_DIR / ".state.json"


def _get_write_dir() -> Path:
    """Return the current subdirectory to write into; state is maintained by the single .state.json file"""
    LOG_DIR.mkdir(exist_ok=True)
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        dir_num = int(state.get("dir", 0))
        count = int(state.get("count", 0))
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        dir_num, count = 0, 0
    if count >= MAX_FILES_PER_DIR:
        dir_num += 1
        count = 0
    target = LOG_DIR / str(dir_num)
    target.mkdir(exist_ok=True)
    with open(_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"dir": dir_num, "count": count + 1}, f)
    return target


def write_log(log_data: dict, request_id: str = None):
    """Write logs to files, split across logs/0, logs/1, ... subdirectories, with at most N files per directory"""
    if request_id is not None and os.environ.get("MODELHUB_LOG", "true").lower()=="true":
        target_dir = _get_write_dir()
        log_file = target_dir / f"{request_id}_{time.time()}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
