"""
Core Audio Processing Module for SanskritMantraPronunciationCoach.

Handles:
- Loading and saving audio (WAV, FLAC preferred; OGG/MP3 with caveats)
- Robust support for extracting audio from video containers (MOV, MP4, MKV, etc.)
  using ffmpeg when available
- Microphone recording with real-time feedback
- Playback
- Feature extraction: MFCC, energy (RMS), pitch (Parselmouth), duration, voicing
- Preprocessing: trim, normalize, resample
- Synthetic reference generation for demo / bootstrapping
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import warnings
import wave
from pathlib import Path
from shutil import which
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import soundfile as sf
import librosa
import parselmouth
from parselmouth.praat import call

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False

from config import AUDIO_CONFIG, REFERENCES_DIR, USER_RECORDINGS_DIR


# =============================================================================
# FORMAT DETECTION & FFMPEG SUPPORT
# =============================================================================

VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".mkv", ".avi", ".webm", ".wmv", ".m4a"}
LOSSY_AUDIO_EXTENSIONS = {".mp3", ".ogg", ".aac", ".wma"}


def _has_ffmpeg() -> bool:
    """Return True if ffmpeg is available on the system PATH."""
    return which("ffmpeg") is not None


def _is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def _is_lossy_audio(path: Path) -> bool:
    return path.suffix.lower() in LOSSY_AUDIO_EXTENSIONS


def extract_audio_from_video(
    video_path: Path | str,
    output_path: Optional[Path] = None,
    sample_rate: Optional[int] = None,
) -> Path:
    """
    Extract the audio track from a video file (MOV, MP4, MKV, AVI, etc.)
    to a temporary WAV file using ffmpeg.

    Returns the path to the extracted WAV file.

    The caller is responsible for cleaning up the returned file if it was
    created as a temporary.
    """
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not _has_ffmpeg():
        raise RuntimeError(
            "ffmpeg is not installed or not found in your system PATH.\n"
            "Please install it to extract audio from video files:\n\n"
            "  macOS:     brew install ffmpeg\n"
            "  Ubuntu:    sudo apt install ffmpeg\n"
            "  Windows:   https://ffmpeg.org/download.html\n"
        )

    if output_path is None:
        fd, output_str = tempfile.mkstemp(suffix=".wav", prefix="sanskrit_extracted_")
        os.close(fd)
        output_path = Path(output_str)
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    sr = sample_rate or AUDIO_CONFIG.sample_rate

    cmd = [
        "ffmpeg",
        "-y",                    # Overwrite without asking
        "-i", str(video_path),
        "-vn",                   # Drop video stream
        "-acodec", "pcm_s16le",  # Uncompressed 16-bit PCM
        "-ar", str(sr),
        "-ac", "1",              # Force mono (good for analysis)
        str(output_path),
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg failed to extract audio from {video_path.name}.\n"
            f"stderr:\n{e.stderr}"
        ) from e

    return output_path


# =============================================================================
# BASIC I/O
# =============================================================================

def load_audio(
    path: Path | str,
    target_sr: Optional[int] = None,
    mono: bool = True,
    allow_video_extraction: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio (or video) file and return (waveform, sample_rate).

    This function provides much better diagnostics than raw librosa.load().

    Features:
    - Clear, actionable error messages when formats are unsupported.
    - Automatic warning when loading lossy formats (.mp3, .ogg, etc.).
    - Optional automatic extraction of audio from video containers
      (MOV, MP4, MKV, AVI, etc.) when ffmpeg is available.

    Args:
        path: Path to audio or video file.
        target_sr: Target sample rate (defaults to config value).
        mono: Convert to mono.
        allow_video_extraction: If True (default), attempt to extract audio
            from video files using ffmpeg if the file looks like a video container.

    Returns:
        (waveform as float32, sample_rate)
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audio/video file not found: {path}")

    target_sr = target_sr or AUDIO_CONFIG.sample_rate
    ext = path.suffix.lower()

    # Warn about lossy formats (especially important for reference files)
    if _is_lossy_audio(path):
        warnings.warn(
            f"Loading lossy format '{ext}': {path.name}\n"
            "For the most accurate pronunciation and prosody analysis, "
            "strongly prefer lossless formats (WAV or FLAC) for reference recordings.",
            UserWarning,
            stacklevel=2,
        )

    # Handle video containers (MOV, MP4, MKV, etc.)
    if _is_video_file(path):
        if not allow_video_extraction:
            raise ValueError(
                f"'{path.name}' appears to be a video file ({ext}).\n"
                "Set allow_video_extraction=True (default) to automatically "
                "extract the audio track using ffmpeg."
            )

        if not _has_ffmpeg():
            raise RuntimeError(
                f"Cannot load video file '{path.name}' ({ext}).\n\n"
                "ffmpeg is required to extract the audio track from video files.\n"
                "Please install it:\n"
                "  • macOS:   brew install ffmpeg\n"
                "  • Ubuntu:  sudo apt install ffmpeg\n"
                "  • Windows: Download from https://ffmpeg.org/download.html\n\n"
                "After installing, re-run your command."
            )

        print(f"🎬 Extracting audio track from video file: {path.name}")
        extracted_wav = extract_audio_from_video(path, sample_rate=target_sr)

        try:
            y, sr = librosa.load(str(extracted_wav), sr=target_sr, mono=mono)
            return y.astype(np.float32), sr
        finally:
            # Clean up the temporary extracted file
            if extracted_wav.exists() and "sanskrit_extracted_" in extracted_wav.name:
                extracted_wav.unlink(missing_ok=True)

    # Standard audio file loading path
    try:
        y, sr = librosa.load(str(path), sr=target_sr, mono=mono)
        return y.astype(np.float32), sr
    except Exception as e:
        supported = "WAV, FLAC (best), OGG (Vorbis), AIFF"
        error_msg = (
            f"Failed to load audio file: {path}\n"
            f"Detected extension: {ext}\n\n"
            f"Error from backend: {e}\n\n"
            f"Directly supported audio formats: {supported}\n"
            f"MP3 support is unreliable and depends on your system libraries.\n\n"
            f"Tip: For reference mantras, use WAV or FLAC for highest quality.\n"
            f"For video files (MOV, MP4, etc.), install ffmpeg and try again."
        )
        raise RuntimeError(error_msg) from e


def save_audio(
    path: Path | str,
    y: np.ndarray,
    sr: int,
    subtype: str = "PCM_16",
) -> Path:
    """
    Save audio array to disk using soundfile.

    Supported output formats include WAV (default), FLAC, and OGG (Vorbis).
    Video containers (MOV, MP4, etc.) are not supported for writing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype=subtype)
    return path


def normalize_audio(y: np.ndarray, headroom_db: float = 0.1) -> np.ndarray:
    """Peak normalize to just under 0 dBFS."""
    peak = np.max(np.abs(y))
    if peak < 1e-9:
        return y
    target = 10 ** (-headroom_db / 20)
    return y * (target / peak)


def trim_silence(
    y: np.ndarray,
    top_db: Optional[int] = None,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Remove leading/trailing silence."""
    top_db = top_db or AUDIO_CONFIG.trim_silence_top_db
    y_trim, _ = librosa.effects.trim(
        y, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )
    return y_trim


# =============================================================================
# FEATURE EXTRACTION (core of pronunciation analysis)
# =============================================================================

def extract_mfcc(
    y: np.ndarray,
    sr: int,
    n_mfcc: Optional[int] = None,
    hop_length: Optional[int] = None,
) -> np.ndarray:
    """Extract MFCC features (transposed for DTW: time x features)."""
    n_mfcc = n_mfcc or AUDIO_CONFIG.n_mfcc
    hop = hop_length or AUDIO_CONFIG.hop_length
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        hop_length=hop,
        n_fft=AUDIO_CONFIG.n_fft,
        win_length=AUDIO_CONFIG.win_length,
    )
    # Delta + delta-delta often helpful for pronunciation
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
    return features.T  # (frames, features)


def extract_rms_energy(
    y: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Short-time RMS energy envelope."""
    rms = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length
    )[0]
    return rms.astype(np.float32)


def extract_pitch_parselmouth(
    y: np.ndarray,
    sr: int,
    time_step: float = 0.01,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 600.0,
) -> Dict[str, np.ndarray]:
    """
    High-quality pitch tracking using Praat via Parselmouth.
    Returns dict with:
      - time: array of time points (seconds)
      - pitch: array of pitch values (Hz), NaN where unvoiced
      - voiced: boolean mask
      - mean_pitch, std_pitch (on voiced regions)
    """
    snd = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=time_step, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    pitch_values = pitch.selected_array["frequency"]  # contains 0 for unvoiced
    pitch_values = np.where(pitch_values == 0, np.nan, pitch_values)

    times = pitch.xs()

    voiced_mask = ~np.isnan(pitch_values)
    voiced_pitches = pitch_values[voiced_mask]

    result = {
        "time": times,
        "pitch": pitch_values,
        "voiced": voiced_mask,
        "mean_pitch": float(np.nanmean(voiced_pitches)) if len(voiced_pitches) > 0 else 0.0,
        "std_pitch": float(np.nanstd(voiced_pitches)) if len(voiced_pitches) > 0 else 0.0,
        "min_pitch": float(np.nanmin(voiced_pitches)) if len(voiced_pitches) > 0 else 0.0,
        "max_pitch": float(np.nanmax(voiced_pitches)) if len(voiced_pitches) > 0 else 0.0,
    }
    return result


def extract_prosodic_features(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Bundle of features useful for Sanskrit mantra evaluation:
    - duration, energy envelope, pitch contour, voicing ratio, spectral centroid
    """
    duration = float(len(y) / sr)

    rms = extract_rms_energy(y, hop_length=AUDIO_CONFIG.hop_length)
    energy_mean = float(np.mean(rms))
    energy_std = float(np.std(rms))
    energy_dyn_range = float(np.max(rms) - np.min(rms)) if len(rms) > 1 else 0.0

    pitch_data = extract_pitch_parselmouth(y, sr)
    pitch_mean = pitch_data["mean_pitch"]
    pitch_std = pitch_data["std_pitch"]
    voiced_ratio = float(np.sum(pitch_data["voiced"]) / max(len(pitch_data["voiced"]), 1))

    # Spectral centroid (brightness)
    cent = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=AUDIO_CONFIG.hop_length
    )[0]
    centroid_mean = float(np.mean(cent))

    # Zero crossing rate (rough noisiness / consonant cue)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=AUDIO_CONFIG.hop_length)[0]
    zcr_mean = float(np.mean(zcr))

    return {
        "duration": duration,
        "energy_mean": energy_mean,
        "energy_std": energy_std,
        "energy_dynamic_range": energy_dyn_range,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "voiced_ratio": voiced_ratio,
        "spectral_centroid_mean": centroid_mean,
        "zcr_mean": zcr_mean,
        "pitch_contour": pitch_data,  # full for detailed comparison
        "rms_envelope": rms,
    }


def compute_dtw_distance(
    seq1: np.ndarray,
    seq2: np.ndarray,
    metric: str = "cosine",
) -> Tuple[float, np.ndarray]:
    """
    Compute DTW distance between two feature sequences (e.g. MFCCs).
    Returns (normalized_distance, alignment_path) where alignment_path has shape (L, 2).
    """
    D, wp = librosa.sequence.dtw(seq1.T, seq2.T, metric=metric)
    raw_dist = D[-1, -1]

    # librosa 0.10+ returns wp with shape (L, 2) already in many cases.
    # Older versions returned (2, L). Normalize to always (L, 2).
    if wp.ndim == 2 and wp.shape[0] == 2 and wp.shape[1] != 2:
        wp = wp.T
    # If it's already (L, 2) or (L, 2) after above, good.

    path_len = wp.shape[0] if wp.ndim == 2 else len(wp)
    norm_dist = raw_dist / max(path_len, 1)

    # Penalize large length differences
    len_penalty = abs(len(seq1) - len(seq2)) / max(len(seq1) + len(seq2), 1)
    final = norm_dist * (1.0 + 0.6 * len_penalty)

    return float(final), wp  # guaranteed (L, 2)


# =============================================================================
# RECORDING & PLAYBACK (pyaudio)
# =============================================================================

class AudioRecorder:
    """Simple blocking recorder with optional max duration and early stop."""

    def __init__(self, config: AudioConfig = AUDIO_CONFIG):
        if not PYAUDIO_AVAILABLE:
            raise RuntimeError(
                "pyaudio is not installed. Install portaudio system library first "
                "(macOS: brew install portaudio) then pip install pyaudio."
            )
        self.config = config
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        self.frames: List[bytes] = []

    def record(
        self,
        max_seconds: Optional[int] = None,
        stop_on_silence: bool = False,
        silence_threshold: float = 0.01,
        silence_duration: float = 2.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Record from default microphone.
        Press Ctrl-C to stop early.
        Returns (audio_float32, sample_rate).
        """
        max_seconds = max_seconds or self.config.record_default_seconds
        sr = self.config.sample_rate
        chunk = self.config.chunk_size
        channels = self.config.channels

        self.stream = self.pyaudio.open(
            format=self.config.format,
            channels=channels,
            rate=sr,
            input=True,
            frames_per_buffer=chunk,
        )

        print("\n🎙️  Recording... (press Ctrl-C to stop early)")
        print("   Speak clearly at a natural chanting pace.\n")

        self.frames = []
        start_time = time.time()
        silent_chunks = 0
        max_silent_chunks = int(silence_duration * sr / chunk)

        try:
            while True:
                data = self.stream.read(chunk, exception_on_overflow=False)
                self.frames.append(data)

                # Simple RMS for silence detection / visual feedback
                audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                rms = np.sqrt(np.mean(audio_chunk**2))

                elapsed = time.time() - start_time
                bar = "█" * int(min(rms * 40, 35))
                print(f"\r   [{elapsed:5.1f}s] {bar:<35} rms={rms:.4f}", end="", flush=True)

                if stop_on_silence:
                    if rms < silence_threshold:
                        silent_chunks += 1
                    else:
                        silent_chunks = 0
                    if silent_chunks > max_silent_chunks and elapsed > 4.0:
                        print("\n   → Auto-stopped on long silence.")
                        break

                if elapsed > max_seconds:
                    print(f"\n   → Max duration ({max_seconds}s) reached.")
                    break

        except KeyboardInterrupt:
            print("\n   → Recording stopped by user.")

        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()

        print("\n✅ Recording complete.\n")

        # Convert to float32 numpy
        audio_bytes = b"".join(self.frames)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        y = audio_int16.astype(np.float32) / 32768.0

        if self.config.normalize:
            y = normalize_audio(y)

        return y, sr

    def close(self):
        if self.stream:
            try:
                self.stream.close()
            except Exception:
                pass
        self.pyaudio.terminate()


def play_audio(y: np.ndarray, sr: int, blocking: bool = True) -> None:
    """Playback using pyaudio. Blocks until done if blocking=True."""
    if not PYAUDIO_AVAILABLE:
        print("⚠️  pyaudio not available for playback. Install it to hear audio.")
        return

    p = pyaudio.PyAudio()
    # Convert float32 -1..1 to int16
    y_int16 = np.clip(y * 32767, -32768, 32767).astype(np.int16)
    data = y_int16.tobytes()

    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sr,
        output=True,
        frames_per_buffer=AUDIO_CONFIG.playback_chunk_size,
    )

    chunk_size = AUDIO_CONFIG.playback_chunk_size * 2  # bytes
    idx = 0
    try:
        while idx < len(data):
            chunk = data[idx : idx + chunk_size]
            stream.write(chunk)
            idx += chunk_size
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


# =============================================================================
# SYNTHETIC REFERENCE GENERATOR (for first-run demo without real recordings)
# =============================================================================

def generate_synthetic_gayatri_reference(
    duration: float = 36.0,
    sr: Optional[int] = None,
    fundamental: float = 138.0,  # comfortable male chanting pitch ~C#3 / D3
) -> np.ndarray:
    """
    Generate a plausible synthetic "reference" chanting of the Gayatri Mantra.
    This is a teaching/demo aid only. Real recorded references are vastly superior.
    Uses additive synthesis + slow pitch and amplitude modulation to mimic Vedic style.
    """
    sr = sr or AUDIO_CONFIG.sample_rate
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Base drone (oṃ-like) + harmonic stack
    y = np.zeros_like(t)

    # Very slow amplitude envelope for the whole mantra (4 logical phrases)
    phrase_boundaries = [0.0, 0.22, 0.48, 0.72, 1.0]
    phrase_amplitudes = [0.95, 0.88, 0.92, 0.85]

    # Simulate the 4 pādas with slight pauses
    pada_durs = [8.5, 9.5, 9.0, 9.0]
    pada_starts = np.cumsum([0] + pada_durs[:-1])
    pada_starts = pada_starts / pada_starts[-1] * duration

    # Main melodic contour (very simple - real chanting has subtle gamaka)
    # Rise slightly on "vareṇyaṃ" and "dhīmahi", settle at end
    pitch_mod = (
        1.0
        + 0.018 * np.sin(2 * np.pi * 0.07 * t)  # gentle vibrato
        + 0.035 * np.sin(2 * np.pi * 0.018 * t)  # slower phrase contour
    )

    # Add the main voiced "chant" component
    for i, (start, dur) in enumerate(zip(pada_starts, pada_durs)):
        mask = (t >= start) & (t < start + dur)
        phase = 2 * np.pi * fundamental * pitch_mod[mask] * (t[mask] - start)
        # Harmonic series (formant-ish for 'a'/'o' vowels)
        component = (
            0.65 * np.sin(phase)
            + 0.28 * np.sin(2 * phase)
            + 0.12 * np.sin(3 * phase)
            + 0.06 * np.sin(4 * phase)
        )
        # Gentle amplitude modulation per phrase
        env = 0.6 + 0.4 * np.sin(np.pi * (t[mask] - start) / dur) ** 0.8
        y[mask] += component * env * phrase_amplitudes[i % len(phrase_amplitudes)]

    # Add soft high-frequency friction for consonants (very light)
    noise = np.random.randn(len(t)) * 0.012
    # High-pass the noise roughly
    noise = librosa.effects.preemphasis(noise, coef=0.95)
    y += noise * (0.3 + 0.7 * (np.abs(y) > 0.08))  # more noise on voiced parts? simplistic

    # Low-pass filter overall (warm chanting tone)
    y = librosa.effects.preemphasis(y, coef=-0.6)  # actually de-emphasis here

    # Final gentle fade in/out
    fade = int(sr * 0.25)
    y[:fade] *= np.linspace(0, 1, fade)
    y[-fade:] *= np.linspace(1, 0, fade)

    y = normalize_audio(y, headroom_db=0.3)
    return y.astype(np.float32)


def ensure_reference_audio(mantra_id: str) -> Optional[Path]:
    """
    Return the path to the reference audio for this mantra, as declared
    in its JSON file (mantras/<id>.json → "reference_audio").

    - If the declared file exists → return its path.
    - If it does not exist:
        - For "gayatri_mantra": generate the built-in synthetic reference (demo only).
        - For any other mantra: print a clear warning and return None.
          The caller is responsible for handling the missing reference.
    """
    from config import MANTRAS_DIR, PROJECT_ROOT

    mantra_json = MANTRAS_DIR / f"{mantra_id}.json"
    if not mantra_json.exists():
        raise FileNotFoundError(f"No mantra definition found: {mantra_json}")

    with open(mantra_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    declared = data.get("reference_audio")
    if not declared:
        # Very old mantra definitions without the field
        declared = f"audio/references/{mantra_id}.wav"

    ref_path = (PROJECT_ROOT / declared).resolve()

    if ref_path.exists():
        return ref_path

    # File is missing → fallback behavior
    if mantra_id == "gayatri_mantra":
        print(f"⚠️  Reference audio not found at {ref_path}")
        print("   Generating synthetic reference for demo purposes...")
        print("   → For accurate evaluation, replace with a real high-quality recording.\n")

        y = generate_synthetic_gayatri_reference()
        save_audio(ref_path, y, AUDIO_CONFIG.sample_rate)
        print(f"   ✓ Synthetic reference saved to {ref_path}\n")
        return ref_path
    else:
        print(f"⚠️  Reference audio not found: {ref_path}")
        print("   Please add a real recording (recommended: WAV or FLAC) at that location.\n")
        return None


# =============================================================================
# UTILITY
# =============================================================================

def get_audio_duration(path: Path | str) -> float:
    y, sr = load_audio(path)
    return len(y) / sr
