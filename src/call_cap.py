# ABOUTME: Tracks and enforces the daily LLM call cap.
# ABOUTME: Counter persists in data/daily_calls.json, resets automatically on date change.

import json
import os
from datetime import date
from pathlib import Path

_CAP_FILE = Path(__file__).parent.parent / "data" / "daily_calls.json"
_DEFAULT_MAX = 50


def _read() -> dict:
    if _CAP_FILE.exists():
        with open(_CAP_FILE) as f:
            return json.load(f)
    return {"date": "", "count": 0}


def _write(data: dict) -> None:
    with open(_CAP_FILE, "w") as f:
        json.dump(data, f)


def check_and_increment() -> bool:
    """Return True if the call is allowed and increment the counter.

    Returns False if the daily cap has been reached.
    """
    max_calls = int(os.getenv("MAX_DAILY_CALLS", _DEFAULT_MAX))
    today = date.today().isoformat()

    data = _read()
    if data["date"] != today:
        data = {"date": today, "count": 0}

    if data["count"] >= max_calls:
        return False

    data["count"] += 1
    _write(data)
    return True


def remaining() -> int:
    """Return how many calls are left today."""
    max_calls = int(os.getenv("MAX_DAILY_CALLS", _DEFAULT_MAX))
    today = date.today().isoformat()
    data = _read()
    if data["date"] != today:
        return max_calls
    return max(0, max_calls - data["count"])
