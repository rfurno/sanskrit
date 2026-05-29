"""
Lightweight helper utilities for SanskritMantraPronunciationCoach.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Optional

from config import MANTRAS_DIR, USER_RECORDINGS_DIR


def load_json(path: Path | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_user_recording(
    y: "np.ndarray",
    sr: int,
    mantra_id: str,
    suffix: str = "",
) -> Path:
    """Save a user attempt with timestamped filename. Returns the saved path."""
    import time
    import numpy as np
    from core.audio_processor import save_audio

    ts = time.strftime("%Y%m%d-%H%M%S")
    safe_suffix = f"_{suffix}" if suffix else ""
    filename = f"{mantra_id}_user_{ts}{safe_suffix}.wav"
    path = USER_RECORDINGS_DIR / filename
    save_audio(path, y, sr)
    return path


def get_latest_user_recording(mantra_id: str) -> Optional[Path]:
    """Return most recent user recording for a mantra, if any."""
    pattern = f"{mantra_id}_user_*.wav"
    candidates = sorted(USER_RECORDINGS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"
